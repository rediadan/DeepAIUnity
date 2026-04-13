using System;
using System.Linq;
using System.Reflection;
using UnityEngine;

namespace DeepAIArena
{
    [Serializable]
    public struct ArenaGhostModelAction
    {
        public float horizontal;
        public bool jump;
        public bool drop;
        public bool shove;
        public string routeName;
    }

    [Serializable]
    public class ArenaNormalizationStats
    {
        public float[] feature_mean;
        public float[] feature_std;
    }

    public class ArenaGhostOnnxPolicy : MonoBehaviour
    {
        [SerializeField] private UnityEngine.Object modelAsset;
        [SerializeField] private TextAsset normalizationStats;
        [SerializeField] private bool preferGpu = true;
        [SerializeField] private float jumpThreshold = 0.5f;
        [SerializeField] private float dropThreshold = 0.5f;
        [SerializeField] private float shoveThreshold = 0.5f;
        [SerializeField] private bool verboseLogging;

        private ArenaNormalizationStats stats;
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
        private int expectedFeatureCount = 19;

        private void OnDisable()
        {
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
                var jumpTensor = peekOutputMethod.Invoke(workerInstance, new object[] { "jump_prob" });
                var dropTensor = peekOutputMethod.Invoke(workerInstance, new object[] { "drop_prob" });
                var shoveTensor = peekOutputMethod.Invoke(workerInstance, new object[] { "shove_prob" });

                var moveLogits = ReadTensor(moveTensor);
                var jumpProb = ReadTensor(jumpTensor);
                var dropProb = ReadTensor(dropTensor);
                var shoveProb = ReadTensor(shoveTensor);

                var moveIndex = ArgMax(moveLogits);
                action = new ArenaGhostModelAction
                {
                    horizontal = moveIndex switch
                    {
                        1 => -1f,
                        2 => 1f,
                        _ => 0f
                    },
                    jump = jumpProb.Length > 0 && jumpProb[0] >= jumpThreshold,
                    drop = dropProb.Length > 0 && dropProb[0] >= dropThreshold,
                    shove = shoveProb.Length > 0 && shoveProb[0] >= shoveThreshold,
                    routeName = "Model"
                };

                return true;
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"ONNX Ghost inference failed: {exception.Message}", this);
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
                Debug.LogWarning($"Failed to initialize ONNX Ghost policy: {exception.Message}", this);
                DisposeWorker();
                return false;
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

            var constructor = tensorShapeType.GetConstructor(new[] { typeof(int), typeof(int), typeof(int), typeof(int) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { 1, 1, 1, featureCount });
            }

            constructor = tensorShapeType.GetConstructor(new[] { typeof(int), typeof(int) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { 1, featureCount });
            }

            constructor = tensorShapeType.GetConstructor(new[] { typeof(int[]) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { new[] { 1, 1, 1, featureCount } });
            }

            return null;
        }

        private float[] BuildNormalizedObservation(ArenaObservationSnapshot observation)
        {
            var values = new[]
            {
                observation.selfPosition.x,
                observation.selfPosition.y,
                observation.opponentPosition.x,
                observation.opponentPosition.y,
                observation.itemPosition.x,
                observation.itemPosition.y,
                observation.selfVelocity.x,
                observation.selfVelocity.y,
                observation.opponentVelocity.x,
                observation.opponentVelocity.y,
                observation.selfBasePosition.x,
                observation.selfBasePosition.y,
                observation.opponentBasePosition.x,
                observation.opponentBasePosition.y,
                observation.selfHasItem ? 1f : 0f,
                observation.opponentHasItem ? 1f : 0f,
                observation.isGrounded ? 1f : 0f,
                observation.opponentGrounded ? 1f : 0f,
                observation.itemLane
            };

            if (values.Length != expectedFeatureCount)
            {
                Debug.LogWarning($"Observation feature count mismatch. expected={expectedFeatureCount}, actual={values.Length}", this);
            }

            LogVerbose($"Observation raw feature count={values.Length}, expectedFeatureCount={expectedFeatureCount}");

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
