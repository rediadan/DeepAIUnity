using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace DeepAIArena
{
    public class ArenaLiveDqnTrainingClient : MonoBehaviour
    {
        private const int ActionCount = 6;
        private const int RasterHeight = ArenaRasterEncoder.RasterHeight;
        private const int RasterWidth = ArenaRasterEncoder.RasterWidth;

        [SerializeField] private ArenaGameManager manager;
        [SerializeField] private ArenaRasterEncoder rasterEncoder;
        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 5055;
        [SerializeField] private float decisionIntervalSeconds = 0.08f;
        [SerializeField] private float actionRequestTimeoutSeconds = 1f;
        [SerializeField] private bool mirrorRightAgent = true;
        [Header("Training Speed")]
        [SerializeField] private bool accelerateTraining = true;
        [SerializeField] private bool accelerateOnlyWhenConnected = true;
        [SerializeField, Range(1f, 50f)] private float trainingTimeScale = 8f;
        [SerializeField] private bool runInBackgroundDuringTraining = true;
        [SerializeField] private bool verboseLogging;

        private readonly ConcurrentQueue<string> outgoingMessages = new();
        private readonly ConcurrentQueue<string> incomingMessages = new();
        private readonly LiveAgentState leftState = new(ArenaSide.Left);
        private readonly LiveAgentState rightState = new(ArenaSide.Right);

        private Thread workerThread;
        private volatile bool workerRunning;
        private volatile bool connected;
        private string helloJson;
        private string lastConnectionStatus = "Disconnected";
        private bool warnedInputMode;
        private float lastEpsilon;
        private int lastGlobalStep;
        private bool hasOriginalTimeSettings;
        private float originalTimeScale = 1f;
        private bool originalRunInBackground;

        private void Awake()
        {
            EnsureReferences();
            helloJson = BuildHelloJson();
        }

        private void OnEnable()
        {
            EnsureReferences();
            helloJson = BuildHelloJson();
            StartWorker();
        }

        private void OnDisable()
        {
            StopWorker();
            RestoreTrainingSpeed();
            ResetAgentState(leftState);
            ResetAgentState(rightState);
        }

        private void OnDestroy()
        {
            StopWorker();
        }

        private void Update()
        {
            EnsureReferences();
            DrainIncomingMessages();

            var leftActive = manager != null && manager.LeftActorMode == ArenaActorControlMode.LiveCnnDqnTraining;
            var rightActive = manager != null && manager.RightActorMode == ArenaActorControlMode.LiveCnnDqnTraining;
            UpdateTrainingSpeed(leftActive || rightActive);
            if (!leftActive && !rightActive)
            {
                return;
            }

            if (!CanEncodeSemanticScreen())
            {
                ApplyIdleIfActive(leftActive, rightActive);
                return;
            }

            if (!connected)
            {
                ResetAgentState(leftState);
                ResetAgentState(rightState);
                ApplyIdleIfActive(leftActive, rightActive);
                return;
            }

            TickAgent(leftState, manager.Player, leftActive);
            TickAgent(rightState, manager.Ghost, rightActive);
        }

        private void EnsureReferences()
        {
            manager ??= GetComponent<ArenaGameManager>();
            rasterEncoder ??= GetComponent<ArenaRasterEncoder>();
        }

        private void OnValidate()
        {
            decisionIntervalSeconds = Mathf.Max(0.02f, decisionIntervalSeconds);
            actionRequestTimeoutSeconds = Mathf.Max(0.1f, actionRequestTimeoutSeconds);
            trainingTimeScale = Mathf.Clamp(trainingTimeScale, 1f, 50f);
        }

        private void UpdateTrainingSpeed(bool liveModeActive)
        {
            if (!Application.isPlaying || !accelerateTraining || !liveModeActive)
            {
                RestoreTrainingSpeed();
                return;
            }

            if (accelerateOnlyWhenConnected && !connected)
            {
                RestoreTrainingSpeed();
                return;
            }

            ApplyTrainingSpeed();
        }

        private void ApplyTrainingSpeed()
        {
            if (!hasOriginalTimeSettings)
            {
                originalTimeScale = Time.timeScale;
                originalRunInBackground = Application.runInBackground;
                hasOriginalTimeSettings = true;
            }

            Time.timeScale = Mathf.Max(1f, trainingTimeScale);
            if (runInBackgroundDuringTraining)
            {
                Application.runInBackground = true;
            }
        }

        private void RestoreTrainingSpeed()
        {
            if (!hasOriginalTimeSettings)
            {
                return;
            }

            Time.timeScale = originalTimeScale;
            Application.runInBackground = originalRunInBackground;
            hasOriginalTimeSettings = false;
        }

        private bool CanEncodeSemanticScreen()
        {
            if (manager == null || rasterEncoder == null)
            {
                return false;
            }

            if (rasterEncoder.InputMode == ArenaRasterInputMode.SemanticRgbScreen
                || rasterEncoder.InputMode == ArenaRasterInputMode.SemanticRgbFrameStack4)
            {
                return true;
            }

            if (!warnedInputMode)
            {
                warnedInputMode = true;
                Debug.LogWarning(
                    "Live CNN-DQN requires ArenaRasterEncoder Input Mode = SemanticRgbFrameStack4 ([12,32,48]) or SemanticRgbScreen ([3,32,48]).",
                    this);
            }

            return false;
        }

        private void ApplyIdleIfActive(bool leftActive, bool rightActive)
        {
            if (leftActive && manager != null && manager.Player != null)
            {
                manager.Player.SetGhostInput(Vector2.zero, false, "LiveCNN-DQN:Waiting");
            }

            if (rightActive && manager != null && manager.Ghost != null)
            {
                manager.Ghost.SetGhostInput(Vector2.zero, false, "LiveCNN-DQN:Waiting");
            }
        }

        private void TickAgent(LiveAgentState state, ArenaCharacterController actor, bool active)
        {
            if (!active || actor == null || manager == null)
            {
                ResetAgentState(state);
                return;
            }

            if (state.roundIndex != manager.RoundIndex)
            {
                ResetAgentState(state);
                state.roundIndex = manager.RoundIndex;
                state.episodeId = manager.RoundIndex;
            }

            if (manager.IsRoundTransitioning)
            {
                SendTerminalTransitionIfNeeded(state);
                actor.SetGhostInput(Vector2.zero, false, "LiveCNN-DQN:RoundDone");
                return;
            }

            if (Time.unscaledTime < state.nextDecisionTime)
            {
                return;
            }

            state.nextDecisionTime = Time.unscaledTime + Mathf.Max(0.02f, decisionIntervalSeconds);
            var currentObservation = manager.BuildObservation(state.side);
            var currentState = EncodeState(state.side);

            if (state.hasPreviousTransition)
            {
                var reward = manager.ConsumeLiveDqnReward(state.side)
                    + manager.ComputeLiveDqnShapingReward(
                        state.previousObservation,
                        currentObservation,
                        state.previousUnityAction);
                EnqueueObserve(
                    state,
                    state.previousState,
                    state.previousModelAction,
                    reward,
                    currentState,
                    false);
            }

            if (state.awaitingActionResponse
                && Time.unscaledTime - state.lastActionRequestTime > actionRequestTimeoutSeconds)
            {
                state.awaitingActionResponse = false;
            }

            if (!state.awaitingActionResponse)
            {
                EnqueueAct(state, currentState);
                state.awaitingActionResponse = true;
                state.lastActionRequestTime = Time.unscaledTime;
            }

            var modelAction = state.hasLatestModelAction ? state.latestModelAction : ArenaDqnAction.Idle;
            var unityAction = ToUnityAction(state.side, modelAction);
            actor.SetGhostInput(ToMove(unityAction), unityAction == ArenaDqnAction.Shove, $"LiveCNN-DQN:{modelAction}");

            state.previousState = currentState;
            state.previousObservation = currentObservation;
            state.previousModelAction = modelAction;
            state.previousUnityAction = unityAction;
            state.hasPreviousTransition = true;
            state.step++;
        }

        private void SendTerminalTransitionIfNeeded(LiveAgentState state)
        {
            if (!state.hasPreviousTransition || state.terminalSent)
            {
                return;
            }

            var currentObservation = manager.BuildObservation(state.side);
            var currentState = EncodeState(state.side);
            var reward = manager.ConsumeLiveDqnReward(state.side)
                + manager.ComputeLiveDqnShapingReward(
                    state.previousObservation,
                    currentObservation,
                    state.previousUnityAction);
            EnqueueObserve(
                state,
                state.previousState,
                state.previousModelAction,
                reward,
                currentState,
                true);
            state.hasPreviousTransition = false;
            state.terminalSent = true;
        }

        private float[] EncodeState(ArenaSide side)
        {
            var state = rasterEncoder.EncodeFlat(side);
            return mirrorRightAgent && side == ArenaSide.Right
                ? MirrorStateX(state, rasterEncoder.InputChannelCount)
                : state;
        }

        private static float[] MirrorStateX(float[] source, int channelCount)
        {
            var mirrored = new float[source.Length];
            var planeSize = RasterHeight * RasterWidth;
            for (var channel = 0; channel < channelCount; channel++)
            {
                var channelOffset = channel * planeSize;
                for (var y = 0; y < RasterHeight; y++)
                {
                    var rowOffset = channelOffset + y * RasterWidth;
                    for (var x = 0; x < RasterWidth; x++)
                    {
                        mirrored[rowOffset + x] = source[rowOffset + (RasterWidth - 1 - x)];
                    }
                }
            }

            return mirrored;
        }

        private ArenaDqnAction ToUnityAction(ArenaSide side, ArenaDqnAction modelAction)
        {
            return mirrorRightAgent && side == ArenaSide.Right ? MirrorActionX(modelAction) : modelAction;
        }

        private static ArenaDqnAction MirrorActionX(ArenaDqnAction action)
        {
            return action switch
            {
                ArenaDqnAction.MoveLeft => ArenaDqnAction.MoveRight,
                ArenaDqnAction.MoveRight => ArenaDqnAction.MoveLeft,
                _ => action
            };
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

        private void EnqueueObserve(
            LiveAgentState state,
            float[] previousState,
            ArenaDqnAction action,
            float reward,
            float[] nextState,
            bool done)
        {
            var message = new ObserveMessage
            {
                agentId = state.side.ToString(),
                episodeId = state.episodeId,
                step = state.step,
                state = previousState,
                action = (int)action,
                reward = reward,
                nextState = nextState,
                done = done
            };
            EnqueueOutgoing(JsonUtility.ToJson(message));
        }

        private void EnqueueAct(LiveAgentState state, float[] currentState)
        {
            var message = new ActMessage
            {
                agentId = state.side.ToString(),
                episodeId = state.episodeId,
                step = state.step,
                state = currentState,
                epsilonAllowed = true
            };
            EnqueueOutgoing(JsonUtility.ToJson(message));
        }

        private void EnqueueOutgoing(string message)
        {
            while (outgoingMessages.Count > 256 && outgoingMessages.TryDequeue(out _))
            {
            }

            outgoingMessages.Enqueue(message);
        }

        private void DrainIncomingMessages()
        {
            while (incomingMessages.TryDequeue(out var raw))
            {
                if (string.IsNullOrWhiteSpace(raw))
                {
                    continue;
                }

                try
                {
                    var action = JsonUtility.FromJson<ActionResponse>(raw);
                    if (action == null || action.type != "action")
                    {
                        continue;
                    }

                    var dqnAction = (ArenaDqnAction)Mathf.Clamp(action.action, 0, ActionCount - 1);
                    var state = action.agentId == ArenaSide.Right.ToString() ? rightState : leftState;
                    state.latestModelAction = dqnAction;
                    state.hasLatestModelAction = true;
                    state.awaitingActionResponse = false;
                    lastEpsilon = action.epsilon;
                    lastGlobalStep = action.globalStep;
                }
                catch (Exception exception)
                {
                    if (verboseLogging)
                    {
                        Debug.LogWarning($"Live CNN-DQN action parse failed: {exception.Message}", this);
                    }
                }
            }
        }

        private string BuildHelloJson()
        {
            var channels = rasterEncoder != null ? rasterEncoder.InputChannelCount : ArenaRasterEncoder.RgbChannelCount;
            var hello = new HelloMessage
            {
                observationFormat = ResolveObservationFormat(),
                shape = new[] { channels, RasterHeight, RasterWidth }
            };
            return JsonUtility.ToJson(hello);
        }

        private string ResolveObservationFormat()
        {
            if (rasterEncoder != null && rasterEncoder.InputMode == ArenaRasterInputMode.SemanticRgbFrameStack4)
            {
                return "semantic_rgb_frame_stack_4";
            }

            return "semantic_rgb";
        }

        private void StartWorker()
        {
            if (workerThread != null)
            {
                return;
            }

            workerRunning = true;
            workerThread = new Thread(NetworkLoop)
            {
                IsBackground = true,
                Name = "ArenaLiveDqnTrainingClient"
            };
            workerThread.Start();
        }

        private void StopWorker()
        {
            workerRunning = false;
            connected = false;
            if (workerThread == null)
            {
                return;
            }

            if (!workerThread.Join(250))
            {
                workerThread.Interrupt();
            }

            workerThread = null;
        }

        private void NetworkLoop()
        {
            while (workerRunning)
            {
                try
                {
                    using var client = new TcpClient();
                    client.NoDelay = true;
                    client.Connect(host, port);
                    using var stream = client.GetStream();
                    using var reader = new StreamReader(stream);
                    using var writer = new StreamWriter(stream) { AutoFlush = true };

                    connected = true;
                    lastConnectionStatus = "Connected";
                    writer.WriteLine(helloJson);

                    while (workerRunning && client.Connected)
                    {
                        while (outgoingMessages.TryDequeue(out var message))
                        {
                            writer.WriteLine(message);
                        }

                        if (stream.DataAvailable)
                        {
                            var line = reader.ReadLine();
                            if (line == null)
                            {
                                break;
                            }

                            incomingMessages.Enqueue(line);
                        }
                        else
                        {
                            Thread.Sleep(2);
                        }
                    }
                }
                catch (ThreadInterruptedException)
                {
                    break;
                }
                catch (Exception exception)
                {
                    lastConnectionStatus = exception.Message;
                    Thread.Sleep(1000);
                }
                finally
                {
                    connected = false;
                }
            }
        }

        private static void ResetAgentState(LiveAgentState state)
        {
            state.hasPreviousTransition = false;
            state.hasLatestModelAction = false;
            state.awaitingActionResponse = false;
            state.latestModelAction = ArenaDqnAction.Idle;
            state.previousModelAction = ArenaDqnAction.Idle;
            state.previousUnityAction = ArenaDqnAction.Idle;
            state.previousState = null;
            state.terminalSent = false;
            state.step = 0;
            state.nextDecisionTime = 0f;
        }

        private void OnGUI()
        {
            if (manager == null
                || (manager.LeftActorMode != ArenaActorControlMode.LiveCnnDqnTraining
                    && manager.RightActorMode != ArenaActorControlMode.LiveCnnDqnTraining))
            {
                return;
            }

            GUILayout.BeginArea(new Rect(12, 545, 460, 90), GUI.skin.box);
            GUILayout.Label($"Live CNN-DQN: {(connected ? "Connected" : "Disconnected")} {lastConnectionStatus}");
            GUILayout.Label($"speed x{Time.timeScale:0.0} / decision {decisionIntervalSeconds:0.000}s real time");
            GUILayout.Label($"epsilon {lastEpsilon:0.000} / global step {lastGlobalStep}");
            GUILayout.Label($"Left {leftState.latestModelAction} / Right {rightState.latestModelAction}");
            GUILayout.EndArea();
        }

        [Serializable]
        private class HelloMessage
        {
            public string type = "hello";
            public int protocolVersion = 1;
            public string envId = "DeepAIArena";
            public string policyId = "ArenaSemanticScreenDqn";
            public string observationFormat = "semantic_rgb";
            public int[] shape = { ArenaRasterEncoder.RgbChannelCount, RasterHeight, RasterWidth };
            public int actionCount = ActionCount;
            public bool sharedPolicy = true;
        }

        [Serializable]
        private class ObserveMessage
        {
            public string type = "observe";
            public string agentId;
            public int episodeId;
            public int step;
            public float[] state;
            public int action;
            public float reward;
            public float[] nextState;
            public bool done;
        }

        [Serializable]
        private class ActMessage
        {
            public string type = "act";
            public string agentId;
            public int episodeId;
            public int step;
            public float[] state;
            public bool epsilonAllowed;
        }

        [Serializable]
        private class ActionResponse
        {
            public string type;
            public string agentId;
            public int action;
            public float epsilon;
            public int globalStep;
        }

        private class LiveAgentState
        {
            public readonly ArenaSide side;
            public int roundIndex = -1;
            public int episodeId;
            public int step;
            public float nextDecisionTime;
            public bool terminalSent;
            public bool awaitingActionResponse;
            public float lastActionRequestTime;
            public bool hasPreviousTransition;
            public float[] previousState;
            public ArenaObservationSnapshot previousObservation;
            public ArenaDqnAction previousModelAction;
            public ArenaDqnAction previousUnityAction;
            public bool hasLatestModelAction;
            public ArenaDqnAction latestModelAction;

            public LiveAgentState(ArenaSide side)
            {
                this.side = side;
            }
        }
    }
}
