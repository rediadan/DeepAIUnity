using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;

namespace DeepAIArena
{
    [Serializable]
    public struct ArenaGhostModelAction
    {
        public Vector2 move;
        public bool shove;
        public string routeName;
    }

    [Serializable]
    public class ArenaNormalizationStats
    {
        public float[] feature_mean;
        public float[] feature_std;
    }

    public enum ArenaModelActorSide
    {
        Auto,
        Left,
        Right
    }

    public class ArenaGhostOnnxPolicy : MonoBehaviour
    {
        private const int BaseFeatureCount = 21;
        private const int V2FeatureCount = 36;
        private const float GuardArrivalDistance = 0.35f;
        private const float GuardAlignmentThreshold = 0.25f;
        private const float SideDividerBypassX = 7.15f;
        private const float LaneCenterGateX = 2.2f;
        private const float CentralPillarOuterX = 1.95f;
        private const float CentralPillarInnerX = 1.05f;
        private const float TopLaneY = 3.05f;
        private const float MiddleApproachLaneY = -2.05f;
        private const float BottomLaneY = -2.3f;
        private const float BaseApproachX = 0.85f;
        private const float GuardWaypointHoldSeconds = 0.28f;
        private const float GuardWaypointReleaseDistance = 0.28f;
        private const float LaneAlignReleaseDistance = 0.7f;
        private const float LaneAlignMaxHoldSeconds = 0.7f;

        [SerializeField] private UnityEngine.Object modelAsset;
        [SerializeField] private TextAsset normalizationStats;
        [SerializeField] private bool preferGpu = true;
        [SerializeField] private float shoveThreshold = 0.5f;
        [SerializeField] private bool verboseLogging;
        [SerializeField] private int sequenceLength = 4;
        [SerializeField] private ArenaModelActorSide modelActorSide = ArenaModelActorSide.Auto;
        [SerializeField] private bool forceObjectiveSteering;

        private ArenaNormalizationStats stats;
        private readonly Queue<float[]> observationHistory = new();
        private object workerInstance;
        private Type tensorType;
        private Type workerType;
        private MethodInfo scheduleMethod;
        private MethodInfo peekOutputMethod;
        private MethodInfo completeMethod;
        private MethodInfo downloadMethod;
        private MethodInfo disposeMethod;
        private bool attemptedInitialization;
        private bool initializationSucceeded;
        private int expectedFeatureCount = BaseFeatureCount;
        private int perFrameFeatureCount = BaseFeatureCount;
        private float lastObservedRoundElapsedTime = -1f;
        private Vector2 heldGuardWaypoint;
        private string heldGuardWaypointName;
        private float heldGuardWaypointUntil = -1f;
        private float heldGuardWaypointStartedAt = -1f;
        private ArenaTargetType heldTargetType;
        private bool hasHeldGuardWaypoint;

        private void OnDisable()
        {
            ResetObservationHistory();
            ResetGuardWaypoint();
            DisposeWorker();
        }

        public bool TryEvaluate(ArenaObservationSnapshot observation, out ArenaGhostModelAction action)
        {
            action = default;

            if (!EnsureInitialized())
            {
                return false;
            }

            var normalized = BuildNormalizedObservation(observation);
            var inputTensor = CreateInputTensor(normalized);
            if (inputTensor == null)
            {
                return false;
            }

            try
            {
                scheduleMethod.Invoke(workerInstance, new[] { inputTensor });

                var moveTensor = peekOutputMethod.Invoke(workerInstance, new object[] { "move_logits" });
                var shoveTensor = peekOutputMethod.Invoke(workerInstance, new object[] { "shove_prob" });

                var moveLogits = ReadTensor(moveTensor);
                var shoveProb = ReadTensor(shoveTensor);

                var moveIndex = ArgMax(moveLogits);
                var isRightSideActor = ShouldMirrorHorizontal(observation);
                var move = ToMoveVector((ArenaMoveAction)moveIndex, isRightSideActor);

                action = new ArenaGhostModelAction
                {
                    move = move,
                    shove = shoveProb.Length > 0 && shoveProb[0] >= shoveThreshold,
                    routeName = "Model"
                };
                action = ApplyTargetDirectionGuard(action, observation, forceObjectiveSteering);

                return true;
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ONNX Ghost inference failed: {DescribeException(exception)}", this);
                DisposeWorker();
                attemptedInitialization = false;
                initializationSucceeded = false;
                return false;
            }
            finally
            {
                DisposeTensor(inputTensor);
            }
        }

        private bool EnsureInitialized()
        {
            if (attemptedInitialization)
            {
                return initializationSucceeded;
            }

            attemptedInitialization = true;
            initializationSucceeded = TryInitializeInternal();
            return initializationSucceeded;
        }

        private bool TryInitializeInternal()
        {
            if (modelAsset == null || normalizationStats == null)
            {
                if (verboseLogging)
                {
                    Debug.LogWarning("Ghost ONNX policy is missing model asset or normalization stats.", this);
                }
                return false;
            }

            stats = JsonUtility.FromJson<ArenaNormalizationStats>(normalizationStats.text);
            if (stats == null || stats.feature_mean == null || stats.feature_std == null)
            {
                Debug.LogWarning("Failed to parse ONNX normalization stats JSON.", this);
                return false;
            }

            if (stats.feature_mean.Length != stats.feature_std.Length || stats.feature_mean.Length == 0)
            {
                Debug.LogWarning("Normalization stats JSON has invalid feature lengths.", this);
                return false;
            }

            expectedFeatureCount = stats.feature_mean.Length;
            if (expectedFeatureCount % V2FeatureCount == 0)
            {
                perFrameFeatureCount = V2FeatureCount;
                sequenceLength = Mathf.Max(1, expectedFeatureCount / V2FeatureCount);
            }
            else if (expectedFeatureCount % BaseFeatureCount == 0)
            {
                perFrameFeatureCount = BaseFeatureCount;
                sequenceLength = Mathf.Max(1, expectedFeatureCount / BaseFeatureCount);
            }

            if (!TryResolveInferenceRuntime(out var runtime))
            {
                if (verboseLogging)
                {
                    Debug.LogWarning("No Unity inference runtime assembly was found. Install com.unity.ai.inference in Package Manager.", this);
                }
                return false;
            }

            return TryCreateWorker(runtime);
        }

        private bool TryCreateWorker(ArenaInferenceRuntime runtime)
        {
            try
            {
                var runtimeModel = LoadRuntimeModel(runtime);
                if (runtimeModel == null)
                {
                    Debug.LogWarning("Failed to load runtime model from ONNX asset.", this);
                    return false;
                }

                LogVerbose(
                    "Resolved inference runtime: "
                    + $"loader={runtime.modelLoaderType?.FullName}, "
                    + $"model={runtime.modelType?.FullName}, "
                    + $"modelAsset={runtime.modelAssetType?.FullName}, "
                    + $"backend={runtime.backendType?.FullName}, "
                    + $"tensor={runtime.tensorFloatType?.FullName}, "
                    + $"workerFactory={runtime.workerFactoryType?.FullName}, "
                    + $"worker={runtime.workerType?.FullName}");

                var backend = Enum.Parse(runtime.backendType, preferGpu ? "GPUCompute" : "CPU");
                workerInstance = CreateWorkerInstance(runtime, runtimeModel, backend);
                if (workerInstance == null)
                {
                    Debug.LogWarning("Failed to create inference worker for ONNX Ghost policy.", this);
                    return false;
                }

                workerType = workerInstance.GetType();
                scheduleMethod = workerType.GetMethod("Schedule", new[] { runtime.tensorFloatType });
                if (scheduleMethod == null)
                {
                    scheduleMethod = workerType.GetMethods().FirstOrDefault(method =>
                    {
                        if (method.Name != "Schedule")
                        {
                            return false;
                        }

                        var parameters = method.GetParameters();
                        return parameters.Length == 1 && parameters[0].ParameterType == runtime.tensorFloatType;
                    });
                }

                peekOutputMethod = workerType.GetMethods().FirstOrDefault(method =>
                {
                    if (method.Name != "PeekOutput")
                    {
                        return false;
                    }

                    var parameters = method.GetParameters();
                    return parameters.Length == 1 && parameters[0].ParameterType == typeof(string);
                });

                if (scheduleMethod == null || peekOutputMethod == null)
                {
                    Debug.LogWarning("The installed Unity inference package does not expose the expected worker API.", this);
                    DisposeWorker();
                    return false;
                }

                tensorType = runtime.tensorFloatType;
                completeMethod = tensorType.GetMethod("CompleteAllPendingOperations")
                    ?? tensorType.GetMethod("CompleteOperationsAndDownload")
                    ?? tensorType.GetMethod("CompleteOperations");
                downloadMethod = tensorType.GetMethod("DownloadToArray")
                    ?? tensorType.GetMethod("ToReadOnlyArray")
                    ?? tensorType.GetMethod("ToArray");
                disposeMethod = tensorType.GetMethod("Dispose");

                LogVerbose(
                    "Initialized worker API: "
                    + $"workerType={workerType?.FullName}, "
                    + $"schedule={scheduleMethod}, "
                    + $"peekOutput={peekOutputMethod}, "
                    + $"complete={completeMethod}, "
                    + $"download={downloadMethod}, "
                    + $"dispose={disposeMethod}");
                return true;
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Failed to initialize ONNX Ghost policy: {DescribeException(exception)}", this);
                DisposeWorker();
                return false;
            }
        }

        public bool TryEvaluateDqn(ArenaObservationSnapshot observation, out ArenaGhostModelAction action)
        {
            action = default;

            if (!EnsureInitialized())
            {
                return false;
            }

            var normalized = BuildNormalizedObservation(observation);
            var inputTensor = CreateInputTensor(normalized);
            if (inputTensor == null)
            {
                return false;
            }

            try
            {
                scheduleMethod.Invoke(workerInstance, new[] { inputTensor });

                var qValueTensor = peekOutputMethod.Invoke(workerInstance, new object[] { "q_values" });
                var qValues = ReadTensor(qValueTensor);
                if (qValues.Length == 0)
                {
                    Debug.LogWarning("DQN ONNX output 'q_values' was empty.", this);
                    return false;
                }

                var actionIndex = ArgMax(qValues);
                var dqnAction = Enum.IsDefined(typeof(ArenaDqnAction), actionIndex)
                    ? (ArenaDqnAction)actionIndex
                    : ArenaDqnAction.Idle;
                var isRightSideActor = ShouldMirrorHorizontal(observation);
                action = ToGhostModelAction(dqnAction, isRightSideActor);
                action = ApplyTargetDirectionGuard(action, observation, forceObjectiveSteering);
                return true;
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"DQN Ghost inference failed: {DescribeException(exception)}", this);
                DisposeWorker();
                attemptedInitialization = false;
                initializationSucceeded = false;
                return false;
            }
            finally
            {
                DisposeTensor(inputTensor);
            }
        }

        private object LoadRuntimeModel(ArenaInferenceRuntime runtime)
        {
            var loadMethod = runtime.modelLoaderType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(method =>
                {
                    if (method.Name != "Load")
                    {
                        return false;
                    }

                    var parameters = method.GetParameters();
                    return parameters.Length == 1 && parameters[0].ParameterType.IsInstanceOfType(modelAsset);
                });

            return loadMethod?.Invoke(null, new[] { modelAsset });
        }

        private object CreateWorkerInstance(ArenaInferenceRuntime runtime, object runtimeModel, object backend)
        {
            if (runtime.workerFactoryType != null)
            {
                var createMethod = runtime.workerFactoryType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .FirstOrDefault(method =>
                    {
                        if (method.Name != "CreateWorker")
                        {
                            return false;
                        }

                        var parameters = method.GetParameters();
                        return parameters.Length == 2
                            && parameters[0].ParameterType == runtime.backendType
                            && parameters[1].ParameterType == runtime.modelType;
                    });

                if (createMethod != null)
                {
                    return createMethod.Invoke(null, new[] { backend, runtimeModel });
                }
            }

            if (runtime.workerType == null)
            {
                return null;
            }

            var constructor = runtime.workerType.GetConstructor(new[] { runtime.modelType, runtime.backendType });
            if (constructor != null)
            {
                return constructor.Invoke(new[] { runtimeModel, backend });
            }

            constructor = runtime.workerType.GetConstructor(new[] { runtime.backendType, runtime.modelType });
            if (constructor != null)
            {
                return constructor.Invoke(new[] { backend, runtimeModel });
            }

            return null;
        }

        private object CreateInputTensor(float[] inputValues)
        {
            var runtime = ResolveRuntimeTypes();
            if (!runtime.isValid)
            {
                return null;
            }

            var shapedValues = EnsureExpectedInputLength(inputValues);
            var tensorShape = CreateTensorShape(runtime.tensorShapeType, shapedValues.Length);
            if (tensorShape == null)
            {
                Debug.LogWarning("Failed to construct inference tensor shape for ONNX Ghost policy.", this);
                return null;
            }

            LogTensorShapeDetails("before-buffer-resize", tensorShape, shapedValues.Length);
            shapedValues = EnsureTensorBufferLength(tensorShape, shapedValues);
            LogTensorShapeDetails("after-buffer-resize", tensorShape, shapedValues.Length);

            var constructor = tensorType.GetConstructor(new[] { runtime.tensorShapeType, typeof(float[]) });
            if (constructor != null)
            {
                LogVerbose($"Using tensor constructor (shape, float[]) with bufferLength={shapedValues.Length}");
                return constructor.Invoke(new object[] { tensorShape, shapedValues });
            }

            constructor = tensorType.GetConstructor(new[] { runtime.tensorShapeType, typeof(float[]), typeof(int) });
            if (constructor != null)
            {
                LogVerbose("Using tensor constructor (shape, float[], int) with dataStartIndex=0");
                return constructor.Invoke(new object[] { tensorShape, shapedValues, 0 });
            }

            Debug.LogWarning("Failed to find a supported tensor constructor for ONNX Ghost policy.", this);
            return null;
        }

        private ArenaRuntimeTypes ResolveRuntimeTypes()
        {
            return new ArenaRuntimeTypes
            {
                tensorShapeType = tensorType?.Assembly.GetType($"{tensorType.Namespace}.TensorShape"),
                isValid = tensorType != null
            };
        }

        private static object CreateTensorShape(Type tensorShapeType, int featureCount)
        {
            if (tensorShapeType == null)
            {
                return null;
            }

            var constructor = tensorShapeType.GetConstructor(new[] { typeof(int), typeof(int) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { 1, featureCount });
            }

            constructor = tensorShapeType.GetConstructor(new[] { typeof(int), typeof(int), typeof(int), typeof(int) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { 1, featureCount, 1, 1 });
            }

            constructor = tensorShapeType.GetConstructor(new[] { typeof(int[]) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { new[] { 1, featureCount } });
            }

            return null;
        }

        private float[] BuildNormalizedObservation(ArenaObservationSnapshot observation)
        {
            var isRightSideActor = ShouldMirrorHorizontal(observation);
            var mirroredSelfPositionX = MirrorX(observation.selfPosition.x, isRightSideActor);
            var mirroredOpponentPositionX = MirrorX(observation.opponentPosition.x, isRightSideActor);
            var mirroredItemPositionX = MirrorX(observation.itemPosition.x, isRightSideActor);
            var mirroredBasePositionX = MirrorX(observation.basePosition.x, isRightSideActor);
            var mirroredTargetPositionX = MirrorX(observation.targetPosition.x, isRightSideActor);
            var mirroredItemDeltaX = MirrorSignedX(observation.itemDelta.x, isRightSideActor);
            var mirroredBaseDeltaX = MirrorSignedX(observation.baseDelta.x, isRightSideActor);
            var mirroredOpponentDeltaX = MirrorSignedX(observation.opponentDelta.x, isRightSideActor);

            var currentFrameValues = BuildFrameValues(
                observation,
                mirroredSelfPositionX,
                mirroredOpponentPositionX,
                mirroredItemPositionX,
                mirroredBasePositionX,
                mirroredTargetPositionX,
                mirroredItemDeltaX,
                mirroredBaseDeltaX,
                mirroredOpponentDeltaX,
                isRightSideActor);

            if (currentFrameValues.Length != perFrameFeatureCount)
            {
                Debug.LogWarning($"Per-frame observation feature count mismatch. expected={perFrameFeatureCount}, actual={currentFrameValues.Length}", this);
            }

            UpdateObservationHistory(currentFrameValues, observation.roundElapsedTime);
            var values = FlattenObservationHistory(currentFrameValues);

            if (values.Length != expectedFeatureCount)
            {
                Debug.LogWarning($"Observation sequence feature count mismatch. expected={expectedFeatureCount}, actual={values.Length}", this);
            }

            LogVerbose($"Observation raw feature count={values.Length}, expectedFeatureCount={expectedFeatureCount}, sequenceLength={sequenceLength}");

            var normalized = new float[values.Length];
            for (var i = 0; i < values.Length; i++)
            {
                var std = i < stats.feature_std.Length ? stats.feature_std[i] : 1f;
                if (Mathf.Abs(std) < 1e-6f)
                {
                    std = 1f;
                }

                var mean = i < stats.feature_mean.Length ? stats.feature_mean[i] : 0f;
                normalized[i] = (values[i] - mean) / std;
            }

            return normalized;
        }

        private float[] BuildFrameValues(
            ArenaObservationSnapshot observation,
            float mirroredSelfPositionX,
            float mirroredOpponentPositionX,
            float mirroredItemPositionX,
            float mirroredBasePositionX,
            float mirroredTargetPositionX,
            float mirroredItemDeltaX,
            float mirroredBaseDeltaX,
            float mirroredOpponentDeltaX,
            bool isRightSideActor)
        {
            var legacyValues = new[]
            {
                mirroredSelfPositionX,
                observation.selfPosition.y,
                mirroredOpponentPositionX,
                observation.opponentPosition.y,
                mirroredItemPositionX,
                observation.itemPosition.y,
                mirroredBasePositionX,
                observation.basePosition.y,
                observation.selfHasItem ? 1f : 0f,
                observation.opponentHasItem ? 1f : 0f,
                mirroredItemDeltaX,
                observation.itemDelta.y,
                mirroredBaseDeltaX,
                observation.baseDelta.y,
                mirroredOpponentDeltaX,
                observation.opponentDelta.y,
                mirroredTargetPositionX,
                observation.targetPosition.y,
                (float)observation.targetType,
                observation.wallAhead ? 1f : 0f,
                observation.roundElapsedTime
            };

            if (perFrameFeatureCount == BaseFeatureCount)
            {
                return legacyValues;
            }

            var values = new float[V2FeatureCount];
            Array.Copy(legacyValues, values, legacyValues.Length);
            var writeIndex = legacyValues.Length;
            values[writeIndex++] = observation.distanceToItem;
            values[writeIndex++] = observation.distanceToBase;
            values[writeIndex++] = observation.distanceToOpponent;
            values[writeIndex++] = observation.distanceToTarget;
            values[writeIndex++] = MirrorSignedX(observation.doorDelta.x, isRightSideActor);
            values[writeIndex++] = observation.doorDelta.y;
            values[writeIndex++] = observation.doorOpen ? 1f : 0f;
            values[writeIndex++] = MirrorSignedX(observation.switchDelta.x, isRightSideActor);
            values[writeIndex++] = observation.switchDelta.y;
            values[writeIndex++] = observation.switchActive ? 1f : 0f;
            values[writeIndex++] = MirrorSignedX(observation.movingObstacleDelta.x, isRightSideActor);
            values[writeIndex++] = observation.movingObstacleDelta.y;
            values[writeIndex++] = MirrorSignedX(observation.movingObstacleVelocity.x, isRightSideActor);
            values[writeIndex++] = observation.movingObstacleVelocity.y;
            values[writeIndex] = observation.movingObstacleAhead ? 1f : 0f;
            return values;
        }

        private float[] EnsureExpectedInputLength(float[] values)
        {
            if (values.Length == expectedFeatureCount)
            {
                return values;
            }

            var resized = new float[expectedFeatureCount];
            var copyLength = Mathf.Min(values.Length, expectedFeatureCount);
            Array.Copy(values, resized, copyLength);
            LogVerbose($"Resized observation buffer from {values.Length} to expectedFeatureCount={expectedFeatureCount}");
            return resized;
        }

        private void UpdateObservationHistory(float[] currentFrameValues, float roundElapsedTime)
        {
            if (roundElapsedTime < lastObservedRoundElapsedTime)
            {
                ResetObservationHistory();
                ResetGuardWaypoint();
            }

            lastObservedRoundElapsedTime = roundElapsedTime;
            observationHistory.Enqueue(currentFrameValues);
            while (observationHistory.Count > sequenceLength)
            {
                observationHistory.Dequeue();
            }
        }

        private float[] FlattenObservationHistory(float[] fallbackFrameValues)
        {
            var frames = observationHistory.ToList();
            if (frames.Count == 0)
            {
                frames.Add(fallbackFrameValues);
            }

            while (frames.Count < sequenceLength)
            {
                frames.Insert(0, frames[0]);
            }

            var flattened = new float[frames.Count * perFrameFeatureCount];
            var writeIndex = 0;
            foreach (var frame in frames)
            {
                Array.Copy(frame, 0, flattened, writeIndex, Mathf.Min(frame.Length, perFrameFeatureCount));
                writeIndex += perFrameFeatureCount;
            }

            return flattened;
        }

        private void ResetObservationHistory()
        {
            observationHistory.Clear();
            lastObservedRoundElapsedTime = -1f;
        }

        private float[] EnsureTensorBufferLength(object tensorShape, float[] values)
        {
            if (tensorShape == null)
            {
                return values;
            }

            var shapeType = tensorShape.GetType();
            var lengthProperty = shapeType.GetProperty("length", BindingFlags.Public | BindingFlags.Instance)
                ?? shapeType.GetProperty("Length", BindingFlags.Public | BindingFlags.Instance);
            var lengthField = shapeType.GetField("length", BindingFlags.Public | BindingFlags.Instance)
                ?? shapeType.GetField("Length", BindingFlags.Public | BindingFlags.Instance);

            var requiredLength = values.Length;
            if (lengthProperty != null)
            {
                requiredLength = Convert.ToInt32(lengthProperty.GetValue(tensorShape));
            }
            else if (lengthField != null)
            {
                requiredLength = Convert.ToInt32(lengthField.GetValue(tensorShape));
            }

            LogVerbose($"Tensor shape requested buffer length={requiredLength}, currentBufferLength={values.Length}");

            if (requiredLength <= values.Length)
            {
                return values;
            }

            var resized = new float[requiredLength];
            Array.Copy(values, resized, values.Length);
            LogVerbose($"Expanded tensor buffer from {values.Length} to runtimeRequiredLength={requiredLength}");
            return resized;
        }

        private void LogTensorShapeDetails(string stage, object tensorShape, int bufferLength)
        {
            if (!verboseLogging || tensorShape == null)
            {
                return;
            }

            var shapeType = tensorShape.GetType();
            var publicMembers = shapeType
                .GetMembers(BindingFlags.Public | BindingFlags.Instance)
                .Where(member => member.MemberType == MemberTypes.Field || member.MemberType == MemberTypes.Property)
                .Select(member =>
                {
                    try
                    {
                        object value = member switch
                        {
                            PropertyInfo property when property.GetIndexParameters().Length == 0 => property.GetValue(tensorShape),
                            FieldInfo field => field.GetValue(tensorShape),
                            _ => null
                        };
                        return $"{member.Name}={value}";
                    }
                    catch
                    {
                        return $"{member.Name}=<unavailable>";
                    }
                });

            Debug.Log(
                $"[ArenaGhostOnnxPolicy] TensorShape {stage}: "
                + $"type={shapeType.FullName}, bufferLength={bufferLength}, values=[{string.Join(", ", publicMembers)}]",
                this);
        }

        private void LogVerbose(string message)
        {
            if (!verboseLogging)
            {
                return;
            }

            Debug.Log($"[ArenaGhostOnnxPolicy] {message}", this);
        }

        private static string DescribeException(Exception exception)
        {
            if (exception == null)
            {
                return "unknown exception";
            }

            var message = $"{exception.GetType().Name}: {exception.Message}";
            var inner = exception.InnerException;
            while (inner != null)
            {
                message += $" | Inner -> {inner.GetType().Name}: {inner.Message}";
                inner = inner.InnerException;
            }

            return message;
        }

        private static float MirrorX(float value, bool shouldMirror)
        {
            return shouldMirror ? -value : value;
        }

        private static float MirrorSignedX(float value, bool shouldMirror)
        {
            return shouldMirror ? -value : value;
        }

        private bool ShouldMirrorHorizontal(ArenaObservationSnapshot observation)
        {
            return modelActorSide switch
            {
                ArenaModelActorSide.Left => false,
                ArenaModelActorSide.Right => true,
                _ => observation.selfPosition.x > 0f
            };
        }

        private float[] ReadTensor(object tensorInstance)
        {
            if (tensorInstance == null)
            {
                return Array.Empty<float>();
            }

            completeMethod?.Invoke(tensorInstance, null);

            if (downloadMethod != null)
            {
                var raw = downloadMethod.Invoke(tensorInstance, null);
                if (raw is float[] values)
                {
                    return values;
                }

                if (raw is Array array)
                {
                    var output = new float[array.Length];
                    for (var i = 0; i < array.Length; i++)
                    {
                        output[i] = Convert.ToSingle(array.GetValue(i));
                    }
                    return output;
                }
            }

            return Array.Empty<float>();
        }

        private void DisposeTensor(object tensorInstance)
        {
            if (tensorInstance == null)
            {
                return;
            }

            disposeMethod?.Invoke(tensorInstance, null);
        }

        private void DisposeWorker()
        {
            if (workerInstance == null)
            {
                return;
            }

            var workerDisposeMethod = workerType?.GetMethod("Dispose");
            workerDisposeMethod?.Invoke(workerInstance, null);
            workerInstance = null;
        }

        private static int ArgMax(float[] values)
        {
            if (values == null || values.Length == 0)
            {
                return 0;
            }

            var bestIndex = 0;
            var bestValue = values[0];
            for (var i = 1; i < values.Length; i++)
            {
                if (values[i] > bestValue)
                {
                    bestValue = values[i];
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static ArenaGhostModelAction ToGhostModelAction(ArenaDqnAction action, bool mirrorHorizontal)
        {
            var modelAction = new ArenaGhostModelAction
            {
                routeName = $"DQN:{action}"
            };

            switch (action)
            {
                case ArenaDqnAction.MoveUp:
                    modelAction.move = Vector2.up;
                    break;
                case ArenaDqnAction.MoveDown:
                    modelAction.move = Vector2.down;
                    break;
                case ArenaDqnAction.MoveLeft:
                    modelAction.move = Vector2.left;
                    break;
                case ArenaDqnAction.MoveRight:
                    modelAction.move = Vector2.right;
                    break;
                case ArenaDqnAction.Shove:
                    modelAction.shove = true;
                    break;
            }

            if (mirrorHorizontal)
            {
                modelAction.move.x *= -1f;
            }

            return modelAction;
        }

        private ArenaGhostModelAction ApplyTargetDirectionGuard(
            ArenaGhostModelAction action,
            ArenaObservationSnapshot observation,
            bool forceObjectiveSteering)
        {
            var waypoint = ResolveStableGuardWaypoint(observation, out var waypointName);
            var targetDelta = waypoint - observation.selfPosition;
            if (targetDelta.magnitude < GuardArrivalDistance)
            {
                return action;
            }

            var shouldForceObjectiveMovement = forceObjectiveSteering
                || observation.selfHasItem
                || observation.opponentHasItem;
            if (!shouldForceObjectiveMovement && action.move.sqrMagnitude < 0.0001f)
            {
                return action;
            }

            var targetDirection = targetDelta.normalized;
            var isAlignedWithWaypoint = Vector2.Dot(action.move.normalized, targetDirection) > GuardAlignmentThreshold;
            var shouldOverrideAlignedMove = observation.wallAhead
                || IsNearCentralPillarChoke(observation.selfPosition, observation.targetPosition);
            if (isAlignedWithWaypoint && !shouldOverrideAlignedMove && !forceObjectiveSteering)
            {
                return action;
            }

            action.move = targetDirection;
            var suffix = shouldForceObjectiveMovement ? ":ObjectiveGuarded" : ":Guarded";
            if (!string.IsNullOrEmpty(waypointName))
            {
                suffix += $"[{waypointName}]";
            }

            action.routeName += suffix;
            return action;
        }

        private Vector2 ResolveStableGuardWaypoint(ArenaObservationSnapshot observation, out string waypointName)
        {
            var resolved = ResolveGuardWaypoint(observation, out var resolvedName);
            if (string.IsNullOrEmpty(resolvedName))
            {
                ResetGuardWaypoint();
                waypointName = resolvedName;
                return resolved;
            }

            var releaseDistance = heldGuardWaypointName == "LaneAlign"
                ? LaneAlignReleaseDistance
                : GuardWaypointReleaseDistance;
            var reachedHeldWaypoint = hasHeldGuardWaypoint
                && Vector2.Distance(observation.selfPosition, heldGuardWaypoint) <= releaseDistance;
            var heldTooLong = hasHeldGuardWaypoint
                && heldGuardWaypointName == "LaneAlign"
                && Time.time - heldGuardWaypointStartedAt >= LaneAlignMaxHoldSeconds;
            var targetTypeChanged = hasHeldGuardWaypoint && heldTargetType != observation.targetType;
            if (hasHeldGuardWaypoint
                && !reachedHeldWaypoint
                && !heldTooLong
                && !targetTypeChanged
                && Time.time < heldGuardWaypointUntil)
            {
                waypointName = heldGuardWaypointName;
                return heldGuardWaypoint;
            }

            var isSameWaypoint = hasHeldGuardWaypoint
                && heldGuardWaypointName == resolvedName
                && Vector2.Distance(heldGuardWaypoint, resolved) <= 0.05f;
            heldGuardWaypoint = resolved;
            heldGuardWaypointName = resolvedName;
            heldGuardWaypointUntil = Time.time + GuardWaypointHoldSeconds;
            if (!isSameWaypoint)
            {
                heldGuardWaypointStartedAt = Time.time;
            }
            heldTargetType = observation.targetType;
            hasHeldGuardWaypoint = true;
            waypointName = resolvedName;
            return resolved;
        }

        private void ResetGuardWaypoint()
        {
            hasHeldGuardWaypoint = false;
            heldGuardWaypoint = Vector2.zero;
            heldGuardWaypointName = string.Empty;
            heldGuardWaypointUntil = -1f;
            heldGuardWaypointStartedAt = -1f;
            heldTargetType = default;
        }

        private static Vector2 ResolveGuardWaypoint(ArenaObservationSnapshot observation, out string waypointName)
        {
            waypointName = string.Empty;

            var self = observation.selfPosition;
            var target = observation.targetPosition;
            var targetDelta = target - self;
            if (targetDelta.magnitude < GuardArrivalDistance)
            {
                return target;
            }

            var laneY = SelectGuardLaneY(observation);
            if (observation.selfHasItem || observation.targetType == ArenaTargetType.Base)
            {
                return ResolveBaseReturnWaypoint(self, target, laneY, out waypointName);
            }

            if (observation.wallAhead)
            {
                waypointName = "WallBypass";
                if (Mathf.Abs(self.y - laneY) > GuardArrivalDistance)
                {
                    return new Vector2(self.x, laneY);
                }

                if (Mathf.Abs(self.x) > LaneCenterGateX)
                {
                    return new Vector2(Mathf.Sign(self.x) * LaneCenterGateX, laneY);
                }

                return new Vector2(0f, laneY);
            }

            if (IsNearCentralPillarChoke(self, target))
            {
                waypointName = "ChokeBypass";
                return new Vector2(Mathf.Sign(self.x) * LaneCenterGateX, laneY);
            }

            if (Mathf.Abs(self.x) > SideDividerBypassX && Mathf.Abs(self.y - laneY) > GuardArrivalDistance)
            {
                waypointName = "SideBypass";
                return new Vector2(self.x, laneY);
            }

            if (ShouldUseLaneWaypoint(observation, laneY))
            {
                if (Mathf.Abs(self.x) > LaneCenterGateX)
                {
                    waypointName = "LaneEntry";
                    return new Vector2(Mathf.Sign(self.x) * LaneCenterGateX, laneY);
                }

                if (!IsCenterTarget(target)
                    && Mathf.Abs(self.y - laneY) > LaneAlignReleaseDistance
                    && Mathf.Abs(target.y - laneY) > GuardArrivalDistance)
                {
                    waypointName = "LaneAlign";
                    return new Vector2(0f, laneY);
                }
            }

            return target;
        }

        private static bool IsCenterTarget(Vector2 target)
        {
            return Mathf.Abs(target.x) < 1.2f && Mathf.Abs(target.y) < 0.8f;
        }

        private static Vector2 ResolveBaseReturnWaypoint(
            Vector2 self,
            Vector2 baseTarget,
            float laneY,
            out string waypointName)
        {
            waypointName = "BaseReturn";

            if (Mathf.Abs(self.x) > LaneCenterGateX)
            {
                waypointName = "BaseLaneEntry";
                return new Vector2(Mathf.Sign(self.x) * LaneCenterGateX, laneY);
            }

            if (Mathf.Abs(self.x) > BaseApproachX)
            {
                waypointName = "BaseCenterAlign";
                return new Vector2(0f, laneY);
            }

            waypointName = "BaseDropIn";
            return baseTarget;
        }

        private static float SelectGuardLaneY(ArenaObservationSnapshot observation)
        {
            if (observation.selfHasItem || observation.targetType == ArenaTargetType.Base)
            {
                return BottomLaneY;
            }

            if (observation.targetPosition.y > 1.5f)
            {
                return TopLaneY;
            }

            if (observation.targetPosition.y < -1f)
            {
                return BottomLaneY;
            }

            return observation.selfPosition.y > 1.4f ? TopLaneY : MiddleApproachLaneY;
        }

        private static bool ShouldUseLaneWaypoint(ArenaObservationSnapshot observation, float laneY)
        {
            var self = observation.selfPosition;
            var target = observation.targetPosition;
            if (observation.selfHasItem || observation.targetType == ArenaTargetType.Base)
            {
                return true;
            }

            if (Mathf.Abs(target.y) < 0.8f)
            {
                return true;
            }

            return Mathf.Abs(self.x) > LaneCenterGateX && Mathf.Abs(self.y - laneY) < 1.25f;
        }

        private static bool IsNearCentralPillarChoke(Vector2 self, Vector2 target)
        {
            if (Mathf.Abs(target.x) > 1.2f || Mathf.Abs(target.y) > 1.1f)
            {
                return false;
            }

            var absX = Mathf.Abs(self.x);
            if (absX < CentralPillarInnerX || absX > CentralPillarOuterX)
            {
                return false;
            }

            return self.y > -1.85f && self.y < 1.95f;
        }

        private static Vector2 ToMoveVector(ArenaMoveAction moveAction, bool mirrorHorizontal)
        {
            var move = moveAction switch
            {
                ArenaMoveAction.MoveUp => Vector2.up,
                ArenaMoveAction.MoveDown => Vector2.down,
                ArenaMoveAction.MoveLeft => Vector2.left,
                ArenaMoveAction.MoveRight => Vector2.right,
                _ => Vector2.zero
            };

            if (mirrorHorizontal)
            {
                move.x *= -1f;
            }

            return move;
        }

        private static bool TryResolveInferenceRuntime(out ArenaInferenceRuntime runtime)
        {
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                var modelLoaderType = assembly.GetType("Unity.Sentis.ModelLoader")
                    ?? assembly.GetType("Unity.InferenceEngine.ModelLoader");
                var modelType = assembly.GetType("Unity.Sentis.Model")
                    ?? assembly.GetType("Unity.InferenceEngine.Model");
                var modelAssetType = assembly.GetType("Unity.Sentis.ModelAsset")
                    ?? assembly.GetType("Unity.InferenceEngine.ModelAsset");
                var backendType = assembly.GetType("Unity.Sentis.BackendType")
                    ?? assembly.GetType("Unity.InferenceEngine.BackendType");
                var tensorFloatType = assembly.GetType("Unity.Sentis.Tensor`1")?.MakeGenericType(typeof(float))
                    ?? assembly.GetType("Unity.InferenceEngine.Tensor`1")?.MakeGenericType(typeof(float));

                if (modelLoaderType == null || modelType == null || modelAssetType == null || backendType == null || tensorFloatType == null)
                {
                    continue;
                }

                runtime = new ArenaInferenceRuntime
                {
                    modelLoaderType = modelLoaderType,
                    modelType = modelType,
                    modelAssetType = modelAssetType,
                    backendType = backendType,
                    tensorFloatType = tensorFloatType,
                    workerFactoryType = assembly.GetType("Unity.Sentis.WorkerFactory")
                        ?? assembly.GetType("Unity.InferenceEngine.WorkerFactory"),
                    workerType = assembly.GetType("Unity.Sentis.Worker")
                        ?? assembly.GetType("Unity.InferenceEngine.Worker")
                };
                return true;
            }

            runtime = default;
            return false;
        }

        private struct ArenaRuntimeTypes
        {
            public Type tensorShapeType;
            public bool isValid;
        }

        private struct ArenaInferenceRuntime
        {
            public Type modelLoaderType;
            public Type modelType;
            public Type modelAssetType;
            public Type backendType;
            public Type tensorFloatType;
            public Type workerFactoryType;
            public Type workerType;
        }
    }
}
