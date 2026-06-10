using System;
using System.Linq;
using System.Reflection;
using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(ArenaCharacterController))]
    public class ArenaRasterOnnxPolicy : MonoBehaviour
    {
        [SerializeField] private ArenaGameManager manager;
        [SerializeField] private ArenaRasterEncoder rasterEncoder;
        [SerializeField] private UnityEngine.Object modelAsset;
        [SerializeField] private string inputName = "state";
        [SerializeField] private string outputName = "q_values";
        [SerializeField] private bool preferGpu = true;
        [SerializeField] private bool autoStep = true;
        [SerializeField] private float stepIntervalSeconds = 0.08f;
        [SerializeField] private bool verboseLogging;

        private ArenaCharacterController controller;
        private object workerInstance;
        private Type tensorType;
        private Type tensorShapeType;
        private Type workerType;
        private MethodInfo scheduleMethod;
        private MethodInfo scheduleWithoutInputMethod;
        private MethodInfo setInputByNameMethod;
        private MethodInfo setInputByIndexMethod;
        private MethodInfo peekOutputMethod;
        private MethodInfo completeMethod;
        private MethodInfo downloadMethod;
        private MethodInfo disposeMethod;
        private bool attemptedInitialization;
        private bool initialized;
        private float nextStepTime;
        private float[] lastQValues = Array.Empty<float>();
        private ArenaDqnAction lastAction;

        private void Reset()
        {
            EnsureDefaultNames();
            controller = GetComponent<ArenaCharacterController>();
            manager = FindAnyObjectByType<ArenaGameManager>();
            rasterEncoder = manager != null ? manager.GetComponent<ArenaRasterEncoder>() : null;
        }

        private void Awake()
        {
            EnsureDefaultNames();
            controller = GetComponent<ArenaCharacterController>();
            if (manager == null) manager = FindAnyObjectByType<ArenaGameManager>();
            if (rasterEncoder == null && manager != null) rasterEncoder = manager.GetComponent<ArenaRasterEncoder>();
        }

        private void OnValidate()
        {
            EnsureDefaultNames();
        }

        private void OnDisable()
        {
            DisposeWorker();
        }

        public void SetAutoStep(bool enabled)
        {
            autoStep = enabled;
        }

        private void Update()
        {
            if (!autoStep || Time.time < nextStepTime)
            {
                return;
            }

            StepOnce();
            nextStepTime = Time.time + stepIntervalSeconds;
        }

        public void StepOnce()
        {
            if (controller == null || manager == null || rasterEncoder == null || manager.IsRoundTransitioning)
            {
                return;
            }

            if (!TryEvaluate(out var action))
            {
                return;
            }

            controller.SetGhostInput(action.move, action.shove, action.routeName);
        }

        public bool TryEvaluate(out ArenaGhostModelAction action)
        {
            action = default;
            if (!TryPredictAction(out var dqnAction))
            {
                return false;
            }

            lastAction = dqnAction;
            action = new ArenaGhostModelAction
            {
                move = ToMove(dqnAction),
                shove = dqnAction == ArenaDqnAction.Shove,
                routeName = $"RasterDQN:{dqnAction}"
            };
            return true;
        }

        public bool TryPredictAction(out ArenaDqnAction action)
        {
            action = ArenaDqnAction.Idle;
            if (!EnsureInitialized())
            {
                return false;
            }

            var inputTensor = CreateInputTensor(rasterEncoder.EncodeFlat(controller.Side));
            if (inputTensor == null)
            {
                return false;
            }

            try
            {
                ScheduleWithInput(inputTensor);
                var outputTensor = peekOutputMethod.Invoke(workerInstance, new object[] { outputName });
                lastQValues = ReadTensor(outputTensor);
                action = (ArenaDqnAction)Mathf.Clamp(ArgMax(lastQValues), 0, 5);
                return true;
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Arena raster ONNX inference failed: {DescribeException(exception)}", this);
                DisposeWorker();
                attemptedInitialization = false;
                initialized = false;
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
                return initialized;
            }

            attemptedInitialization = true;
            initialized = TryInitialize();
            return initialized;
        }

        private bool TryInitialize()
        {
            EnsureDefaultNames();
            if (modelAsset == null)
            {
                Debug.LogWarning("ArenaRasterOnnxPolicy has no model asset assigned.", this);
                return false;
            }

            if (!TryResolveInferenceRuntime(out var runtime))
            {
                Debug.LogWarning("No Unity Sentis/InferenceEngine runtime was found.", this);
                return false;
            }

            var runtimeModel = LoadRuntimeModel(runtime);
            if (runtimeModel == null)
            {
                Debug.LogWarning("Failed to load Arena raster ONNX model asset.", this);
                return false;
            }

            tensorType = runtime.tensorFloatType;
            tensorShapeType = tensorType.Assembly.GetType($"{tensorType.Namespace}.TensorShape");
            var backend = Enum.Parse(runtime.backendType, preferGpu ? "GPUCompute" : "CPU");
            workerInstance = CreateWorkerInstance(runtime, runtimeModel, backend);
            if (workerInstance == null)
            {
                Debug.LogWarning("Failed to create Sentis/InferenceEngine worker.", this);
                return false;
            }

            workerType = workerInstance.GetType();
            scheduleMethod = FindScheduleMethod(workerType, runtime.tensorFloatType);
            scheduleWithoutInputMethod = workerType.GetMethod("Schedule", Type.EmptyTypes);
            setInputByNameMethod = FindSetInputByNameMethod(workerType, runtime.tensorFloatType);
            setInputByIndexMethod = FindSetInputByIndexMethod(workerType, runtime.tensorFloatType);
            peekOutputMethod = FindPeekOutputMethod(workerType);
            completeMethod = tensorType.GetMethod("CompleteAllPendingOperations")
                ?? tensorType.GetMethod("CompleteOperationsAndDownload");
            downloadMethod = tensorType.GetMethod("DownloadToArray")
                ?? tensorType.GetMethod("ToReadOnlyArray");
            disposeMethod = tensorType.GetMethod("Dispose");

            var canSchedule = scheduleMethod != null
                || (scheduleWithoutInputMethod != null && (setInputByNameMethod != null || setInputByIndexMethod != null));
            var ok = canSchedule && peekOutputMethod != null && tensorShapeType != null;
            if (verboseLogging)
            {
                Debug.Log(
                    $"[ArenaRasterOnnxPolicy] initialized={ok}, inputName={inputName}, "
                    + $"input=[1,{InputChannelCount},{ArenaRasterEncoder.RasterHeight},{ArenaRasterEncoder.RasterWidth}], "
                    + $"inputMode={rasterEncoder.InputMode}, "
                    + $"setInputByName={setInputByNameMethod != null}, setInputByIndex={setInputByIndexMethod != null}, "
                    + $"scheduleNoInput={scheduleWithoutInputMethod != null}, scheduleTensor={scheduleMethod != null}",
                    this);
            }

            return ok;
        }

        private void EnsureDefaultNames()
        {
            if (string.IsNullOrWhiteSpace(inputName))
            {
                inputName = "state";
            }

            if (string.IsNullOrWhiteSpace(outputName))
            {
                outputName = "q_values";
            }
        }

        private object LoadRuntimeModel(GridInferenceRuntime runtime)
        {
            var loadMethod = runtime.modelLoaderType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(method =>
                {
                    if (method.Name != "Load")
                    {
                        return false;
                    }

                    var parameters = method.GetParameters();
                    return parameters.Length == 1 && parameters[0].ParameterType.IsAssignableFrom(modelAsset.GetType());
                });

            return loadMethod?.Invoke(null, new[] { modelAsset });
        }

        private object CreateInputTensor(float[] values)
        {
            var shape = CreateTensorShape();
            if (shape == null)
            {
                return null;
            }

            var constructor = tensorType.GetConstructor(new[] { tensorShapeType, typeof(float[]) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { shape, values });
            }

            constructor = tensorType.GetConstructor(new[] { tensorShapeType, typeof(float[]), typeof(int) });
            return constructor?.Invoke(new object[] { shape, values, 0 });
        }

        private object CreateTensorShape()
        {
            var constructor = tensorShapeType.GetConstructor(new[] { typeof(int), typeof(int), typeof(int), typeof(int) });
            if (constructor != null)
            {
                return constructor.Invoke(new object[] { 1, InputChannelCount, ArenaRasterEncoder.RasterHeight, ArenaRasterEncoder.RasterWidth });
            }

            constructor = tensorShapeType.GetConstructor(new[] { typeof(int[]) });
            return constructor?.Invoke(new object[] { new[] { 1, InputChannelCount, ArenaRasterEncoder.RasterHeight, ArenaRasterEncoder.RasterWidth } });
        }

        private int InputChannelCount => rasterEncoder != null ? rasterEncoder.InputChannelCount : ArenaRasterEncoder.ChannelCount;

        private static MethodInfo FindScheduleMethod(Type worker, Type tensor)
        {
            return worker.GetMethods().FirstOrDefault(method =>
            {
                if (method.Name != "Schedule")
                {
                    return false;
                }

                var parameters = method.GetParameters();
                return parameters.Length == 1 && parameters[0].ParameterType.IsAssignableFrom(tensor);
            });
        }

        private static MethodInfo FindSetInputByNameMethod(Type worker, Type tensor)
        {
            return worker.GetMethods().FirstOrDefault(method =>
            {
                if (method.Name != "SetInput")
                {
                    return false;
                }

                var parameters = method.GetParameters();
                return parameters.Length == 2
                    && parameters[0].ParameterType == typeof(string)
                    && parameters[1].ParameterType.IsAssignableFrom(tensor);
            });
        }

        private static MethodInfo FindSetInputByIndexMethod(Type worker, Type tensor)
        {
            return worker.GetMethods().FirstOrDefault(method =>
            {
                if (method.Name != "SetInput")
                {
                    return false;
                }

                var parameters = method.GetParameters();
                return parameters.Length == 2
                    && parameters[0].ParameterType == typeof(int)
                    && parameters[1].ParameterType.IsAssignableFrom(tensor);
            });
        }

        private static MethodInfo FindPeekOutputMethod(Type worker)
        {
            return worker.GetMethods().FirstOrDefault(method =>
            {
                if (method.Name != "PeekOutput")
                {
                    return false;
                }

                var parameters = method.GetParameters();
                return parameters.Length == 1 && parameters[0].ParameterType == typeof(string);
            });
        }

        private object CreateWorkerInstance(GridInferenceRuntime runtime, object runtimeModel, object backend)
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

            var constructor = runtime.workerType?.GetConstructor(new[] { runtime.modelType, runtime.backendType })
                ?? runtime.workerType?.GetConstructor(new[] { runtime.backendType, runtime.modelType });
            return constructor?.Invoke(constructor.GetParameters()[0].ParameterType == runtime.modelType
                ? new[] { runtimeModel, backend }
                : new[] { backend, runtimeModel });
        }

        private void ScheduleWithInput(object inputTensor)
        {
            if (scheduleWithoutInputMethod != null)
            {
                if (setInputByNameMethod != null && !string.IsNullOrWhiteSpace(inputName))
                {
                    setInputByNameMethod.Invoke(workerInstance, new[] { inputName, inputTensor });
                }
                else if (setInputByIndexMethod != null)
                {
                    setInputByIndexMethod.Invoke(workerInstance, new[] { 0, inputTensor });
                }
                else if (scheduleMethod != null)
                {
                    scheduleMethod.Invoke(workerInstance, new[] { inputTensor });
                    return;
                }

                scheduleWithoutInputMethod.Invoke(workerInstance, null);
                return;
            }

            scheduleMethod.Invoke(workerInstance, new[] { inputTensor });
        }

        private float[] ReadTensor(object tensorInstance)
        {
            if (tensorInstance == null)
            {
                return Array.Empty<float>();
            }

            completeMethod?.Invoke(tensorInstance, null);
            if (downloadMethod == null)
            {
                return Array.Empty<float>();
            }

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

            return Array.Empty<float>();
        }

        private void DisposeTensor(object tensorInstance)
        {
            disposeMethod?.Invoke(tensorInstance, null);
        }

        private void DisposeWorker()
        {
            if (workerInstance == null)
            {
                return;
            }

            workerType?.GetMethod("Dispose")?.Invoke(workerInstance, null);
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

        private static Vector2 ToMove(ArenaDqnAction action)
        {
            return action switch
            {
                ArenaDqnAction.MoveUp => Vector2.up,
                ArenaDqnAction.MoveDown => Vector2.down,
                ArenaDqnAction.MoveLeft => Vector2.left,
                ArenaDqnAction.MoveRight => Vector2.right,
                _ => Vector2.zero
            };
        }

        private static bool TryResolveInferenceRuntime(out GridInferenceRuntime runtime)
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

                runtime = new GridInferenceRuntime
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

        private static string DescribeException(Exception exception)
        {
            var message = $"{exception.GetType().Name}: {exception.Message}";
            var inner = exception.InnerException;
            while (inner != null)
            {
                message += $" | Inner -> {inner.GetType().Name}: {inner.Message}";
                inner = inner.InnerException;
            }

            return message;
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(12, 430, 420, 105), GUI.skin.box);
            GUILayout.Label($"Arena Raster ONNX action: {lastAction}");
            GUILayout.Label($"Input: [1,{InputChannelCount},{ArenaRasterEncoder.RasterHeight},{ArenaRasterEncoder.RasterWidth}] ({rasterEncoder?.InputMode})");
            GUILayout.Label($"Q: {string.Join(", ", lastQValues.Select(value => value.ToString("0.000")))}");
            GUILayout.EndArea();
        }

        private struct GridInferenceRuntime
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
