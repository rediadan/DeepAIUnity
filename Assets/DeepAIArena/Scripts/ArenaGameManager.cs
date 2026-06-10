using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepAIArena
{
    public enum ArenaSide
    {
        Left,
        Right
    }

    public enum ArenaMoveAction
    {
        Idle,
        MoveUp,
        MoveDown,
        MoveLeft,
        MoveRight
    }

    public enum ArenaRewardEventType
    {
        RoundStart,
        ItemCollected,
        ItemDelivered,
        ItemLost,
        FellOffMap,
        RoundTimeout,
        OpenDoorSwitchPressed
    }

    public enum ArenaTargetType
    {
        Item = 0,
        Base = 1,
        Opponent = 2
    }

    public enum ArenaDqnAction
    {
        Idle = 0,
        MoveUp = 1,
        MoveDown = 2,
        MoveLeft = 3,
        MoveRight = 4,
        Shove = 5
    }

    public enum ArenaActorControlMode
    {
        Human,
        RuleBased,
        OnnxInference,
        DqnInference,
        RasterDqnInference,
        MlAgents,
        LiveCnnDqnTraining
    }

    public enum ArenaItemSpawnMode
    {
        Fixed,
        Cycle,
        Random
    }

    [System.Serializable]
    public struct ArenaObservationSnapshot
    {
        public Vector2 selfPosition;
        public Vector2 opponentPosition;
        public Vector2 itemPosition;
        public Vector2 basePosition;
        public bool selfHasItem;
        public bool opponentHasItem;
        public Vector2 itemDelta;
        public Vector2 baseDelta;
        public Vector2 opponentDelta;
        public float distanceToItem;
        public float distanceToBase;
        public float distanceToOpponent;
        public float distanceToTarget;
        public Vector2 targetPosition;
        public ArenaTargetType targetType;
        public bool wallAhead;
        public float roundElapsedTime;
        public Vector2 doorDelta;
        public bool doorOpen;
        public Vector2 switchDelta;
        public bool switchActive;
        public Vector2 movingObstacleDelta;
        public Vector2 movingObstacleVelocity;
        public bool movingObstacleAhead;
        public bool shortcutBlocked;
        public bool detourNeeded;
    }

    [System.Serializable]
    public struct ArenaActionSnapshot
    {
        public ArenaMoveAction moveAction;
        public bool shovePressed;
        public string routeName;
    }

    [System.Serializable]
    public struct ArenaRewardEvent
    {
        public ArenaRewardEventType eventType;
        public ArenaSide actorSide;
        public float rewardDelta;
        public float timestamp;
        public int roundIndex;
        public string note;
    }

    [System.Serializable]
    public struct ArenaDqnTransitionLog
    {
        public string transitionType;
        public string actorSide;
        public int roundIndex;
        public float timestamp;
        public ArenaObservationSnapshot state;
        public ArenaDqnAction action;
        public float reward;
        public ArenaObservationSnapshot nextState;
        public bool done;
    }

    [System.Serializable]
    public struct ArenaCausalStepResult
    {
        public bool pickedItem;
        public bool scored;
        public bool droppedItem;
        public bool shoveAttempted;
        public bool shoveSucceeded;
        public bool forcedItemDrop;
        public float reward;
        public bool done;
    }

    [System.Serializable]
    public struct ArenaCausalStepLog
    {
        public string logType;
        public string actorSide;
        public int roundIndex;
        public float timestamp;
        public ArenaObservationSnapshot observation;
        public ArenaActionSnapshot action;
        public ArenaCausalStepResult result;
        public string causalTags;
    }

    [System.Serializable]
    public struct ArenaRoundSummaryLog
    {
        public string logType;
        public int roundIndex;
        public float duration;
        public string outcome;
        public int leftScore;
        public int rightScore;
        public ArenaRoundActorStats left;
        public ArenaRoundActorStats right;
        public string itemSpawnMode;
        public int itemSpawnIndex;
        public int doorOpenCount;
        public int switchActivationCount;
    }

    [System.Serializable]
    public struct ArenaRoundActorStats
    {
        public string actorSide;
        public int scoreAtEnd;
        public int itemCollectedCount;
        public int itemDeliveredCount;
        public int doorOpenCount;
        public int switchActivationCount;
        public int shortcutBlockedCount;
        public int shoveAttemptCount;
        public int shoveSuccessCount;
        public int forcedItemDropCount;
        public float wallBlockedTime;
        public float movingObstacleBlockedTime;
        public float idleTime;
    }

    public class ArenaGameManager : MonoBehaviour
    {
        private readonly Vector3[] itemSpawnPoints =
        {
            new Vector3(0f, 0f, 0f),
            new Vector3(0f, 3.25f, 0f),
            new Vector3(0f, -2.3f, 0f)
        };

        [SerializeField] private ArenaCharacterController player;
        [SerializeField] private ArenaCharacterController ghost;
        [SerializeField] private ArenaItem item;
        [SerializeField] private ArenaBaseZone sharedBase;
        [SerializeField] private ArenaDoor[] arenaDoors;
        [SerializeField] private ArenaSwitch[] arenaSwitches;
        [SerializeField] private ArenaMovingObstacle[] movingObstacles;
        [Header("Item Spawn Curriculum")]
        [SerializeField] private ArenaItemSpawnMode itemSpawnMode = ArenaItemSpawnMode.Fixed;
        [SerializeField] private int fixedItemSpawnIndex;
        [Header("Return-To-Base Curriculum")]
        [SerializeField] private bool enableReturnToBaseCurriculum;
        [SerializeField, Range(0f, 1f)] private float returnToBaseCurriculumChance = 0.35f;
        [SerializeField] private int returnToBaseCurriculumUntilRound;
        [SerializeField] private bool alternateReturnToBaseHolder = true;
        [SerializeField] private Vector2 returnToBaseLeftStart = new(-1.6f, -0.15f);
        [SerializeField] private Vector2 returnToBaseRightStart = new(1.6f, -0.15f);
        [Header("Round Rules")]
        [SerializeField] private float roundDurationSeconds = 60f;
        [SerializeField] private float roundResetDelaySeconds = 1.25f;
        [Header("Actor Control")]
        [SerializeField] private ArenaActorControlMode leftActorMode = ArenaActorControlMode.Human;
        [SerializeField] private ArenaActorControlMode rightActorMode = ArenaActorControlMode.DqnInference;
        [Header("Logging")]
        [SerializeField] private bool writeLogsToFile = true;
        [SerializeField] private string outputDirectoryName = "arena_training_rounds";
        [SerializeField] private string evaluationSummaryFileName = "arena_round_summary.jsonl";
        [SerializeField] private bool writeDqnTransitions = true;
        [Tooltip("Minimum seconds between Step/CausalStep logs. Set 0 to log every frame.")]
        [SerializeField, Min(0f)] private float stepLogIntervalSeconds = 0.1f;
        [Tooltip("Minimum seconds between DqnTransition logs. Set 0 to log every frame.")]
        [SerializeField, Min(0f)] private float dqnTransitionLogIntervalSeconds = 0.1f;
        [Header("Rewards")]
        [SerializeField] private float itemPickupReward = 1f;
        [SerializeField] private float timeoutNoItemPenalty = -0.5f;
        [SerializeField] private float dqnStepPenalty = -0.01f;
        [SerializeField] private float dqnTargetProgressRewardScale = 0.05f;
        [SerializeField] private float dqnItemTargetProgressRewardScale = 0.08f;
        [SerializeField] private float dqnBaseTargetProgressRewardScale = 0.2f;
        [SerializeField] private float dqnOpponentTargetProgressRewardScale = 0.05f;
        [SerializeField] private float dqnProgressRewardClamp = 0.25f;
        [SerializeField] private float dqnWallActionPenalty = -0.1f;
        [SerializeField] private float dqnInvalidShovePenalty = -0.03f;
        [SerializeField] private float dqnShoveOpportunityDistance = 0.95f;
        [Header("Path Guidance Rewards")]
        [SerializeField] private bool usePathDistanceReward = true;
        [SerializeField] private bool useFlowFieldDirectionReward = true;
        [SerializeField] private Vector2 pathWorldMin = new(-10.4f, -5.6f);
        [SerializeField] private Vector2 pathWorldMax = new(10.4f, 5.6f);
        [SerializeField, Range(16, 96)] private int pathGridWidth = 48;
        [SerializeField, Range(12, 64)] private int pathGridHeight = 32;
        [SerializeField] private float pathDistanceRewardScale = 0.025f;
        [SerializeField] private float pathFlowDirectionReward = 0.008f;
        [SerializeField] private float pathFlowDirectionPenalty = -0.004f;
        [SerializeField] private bool pathTreatClosedDoorsAsBlocked = true;
        [SerializeField] private bool pathTreatMovingObstaclesAsBlocked;
        [SerializeField] private float mlDoorOpenReward = 0.3f;
        [SerializeField] private float openDoorSwitchPenalty = -0.15f;
        [SerializeField] private float mlClosedDoorActionPenalty = -0.015f;
        [SerializeField] private float mlMovingObstacleActionPenalty = -0.03f;

        private int leftScore;
        private int rightScore;
        private int roundIndex;
        private bool roundTransition;
        private float roundStartTime;
        private string outputDirectoryPath;
        private string currentRoundOutputPath;
        private string evaluationSummaryOutputPath;
        private float pendingLeftDqnReward;
        private float pendingRightDqnReward;
        private float pendingLeftMlAgentReward;
        private float pendingRightMlAgentReward;
        private float pendingLeftLiveDqnReward;
        private float pendingRightLiveDqnReward;
        private ArenaCausalStepResult pendingLeftCausalResult;
        private ArenaCausalStepResult pendingRightCausalResult;
        private ArenaRoundActorStats leftRoundStats;
        private ArenaRoundActorStats rightRoundStats;
        private int roundDoorOpenCount;
        private int roundSwitchActivationCount;
        private int currentItemSpawnIndex = -1;
        private int nextCycleItemSpawnIndex;
        private bool hasPreviousPlayerDqnStep;
        private ArenaObservationSnapshot previousPlayerObservation;
        private ArenaActionSnapshot previousPlayerAction;
        private float nextLeftStepLogTime;
        private float nextRightStepLogTime;
        private float nextPlayerDqnTransitionLogTime;

        public ArenaCharacterController Player => player;
        public ArenaCharacterController Ghost => ghost;
        public ArenaItem Item => item;
        public float ItemPickupReward => itemPickupReward;
        public ArenaActorControlMode LeftActorMode => leftActorMode;
        public ArenaActorControlMode RightActorMode => rightActorMode;
        public ArenaDoor[] ArenaDoors => arenaDoors;
        public ArenaSwitch[] ArenaSwitches => arenaSwitches;
        public ArenaMovingObstacle[] MovingObstacles => movingObstacles;
        public Vector3[] ItemSpawnPoints => itemSpawnPoints;
        public ArenaItemSpawnMode ItemSpawnMode => itemSpawnMode;
        public int CurrentItemSpawnIndex => currentItemSpawnIndex;
        public ArenaSide LastScoringSide { get; private set; }
        public int RoundIndex => roundIndex;
        public float RoundElapsedTime => Time.time - roundStartTime;
        public bool IsRoundTransitioning => roundTransition;

        private void Awake()
        {
            EnsureReferences();
            outputDirectoryPath = Path.Combine(Application.persistentDataPath, outputDirectoryName);
            if (writeLogsToFile && Directory.Exists(outputDirectoryPath))
            {
                Directory.Delete(outputDirectoryPath, true);
            }

            if (writeLogsToFile)
            {
                Directory.CreateDirectory(outputDirectoryPath);
                evaluationSummaryOutputPath = Path.Combine(outputDirectoryPath, evaluationSummaryFileName);
                if (File.Exists(evaluationSummaryOutputPath))
                {
                    File.Delete(evaluationSummaryOutputPath);
                }
            }

            if (HasRequiredReferences())
            {
                InitializeRuntimeReferences();
            }
        }

        private void OnValidate()
        {
            EnsureReferences();
        }

        public void Configure(
            ArenaCharacterController playerController,
            ArenaCharacterController ghostController,
            ArenaItem arenaItem,
            ArenaBaseZone arenaSharedBase)
        {
            player = playerController;
            ghost = ghostController;
            item = arenaItem;
            sharedBase = arenaSharedBase;

            player.Initialize(this);
            ghost.Initialize(this);
            arenaSharedBase.Initialize(this);
            item.Initialize(this);
            EnsureReferences();
            InitializeEnvironmentObjects();
            ApplyActorControlModes();

            BeginRound();
        }

        public void SetActorControlModes(ArenaActorControlMode leftMode, ArenaActorControlMode rightMode)
        {
            leftActorMode = leftMode;
            rightActorMode = rightMode;
            ApplyActorControlModes();
        }

        private void EnsureReferences()
        {
            if (player == null || ghost == null)
            {
                var controllers = GetComponentsInChildren<ArenaCharacterController>(true);
                foreach (var controller in controllers)
                {
                    if (controller == null)
                    {
                        continue;
                    }

                    if (controller.Side == ArenaSide.Left)
                    {
                        player ??= controller;
                    }
                    else if (controller.Side == ArenaSide.Right)
                    {
                        ghost ??= controller;
                    }
                }
            }

            item ??= GetComponentInChildren<ArenaItem>(true);
            if (arenaDoors == null || arenaDoors.Length == 0)
            {
                arenaDoors = GetArenaComponentsInScope<ArenaDoor>();
            }

            if (arenaSwitches == null || arenaSwitches.Length == 0)
            {
                arenaSwitches = GetArenaComponentsInScope<ArenaSwitch>();
            }

            if (movingObstacles == null || movingObstacles.Length == 0)
            {
                movingObstacles = GetArenaComponentsInScope<ArenaMovingObstacle>();
            }

            if (sharedBase == null)
            {
                var bases = GetComponentsInChildren<ArenaBaseZone>(true);
                foreach (var arenaBase in bases)
                {
                    if (arenaBase == null)
                    {
                        continue;
                    }

                    if (arenaBase.SharedBase)
                    {
                        sharedBase ??= arenaBase;
                    }
                }
            }
        }

        private T[] GetArenaComponentsInScope<T>() where T : Component
        {
            var scope = transform.parent != null ? transform.parent : transform;
            return scope.GetComponentsInChildren<T>(true);
        }

        private bool HasRequiredReferences()
        {
            return player != null && ghost != null && item != null && sharedBase != null;
        }

        private void InitializeRuntimeReferences()
        {
            player.Initialize(this);
            ghost.Initialize(this);
            sharedBase.Initialize(this);
            item.Initialize(this);
            InitializeEnvironmentObjects();
            ApplyActorControlModes();
        }

        private void InitializeEnvironmentObjects()
        {
            if (arenaDoors != null)
            {
                foreach (var arenaDoor in arenaDoors)
                {
                    if (arenaDoor != null)
                    {
                        arenaDoor.Initialize(this);
                    }
                }
            }

            if (arenaSwitches != null)
            {
                foreach (var arenaSwitch in arenaSwitches)
                {
                    if (arenaSwitch != null)
                    {
                        arenaSwitch.Initialize(this);
                    }
                }
            }
        }

        public void RefreshEnvironmentReferences()
        {
            arenaDoors = GetArenaComponentsInScope<ArenaDoor>();
            arenaSwitches = GetArenaComponentsInScope<ArenaSwitch>();
            movingObstacles = GetArenaComponentsInScope<ArenaMovingObstacle>();
            InitializeEnvironmentObjects();
        }

        private void EnsureEnvironmentReferencesAvailable()
        {
            if ((arenaDoors == null || arenaDoors.Length == 0)
                && (arenaSwitches == null || arenaSwitches.Length == 0)
                && (movingObstacles == null || movingObstacles.Length == 0))
            {
                RefreshEnvironmentReferences();
            }
        }

        private void ApplyActorControlModes()
        {
            ConfigureActorControl(player, leftActorMode);
            ConfigureActorControl(ghost, rightActorMode);
            ConfigureLiveDqnTrainingClient();
        }

        private void ConfigureLiveDqnTrainingClient()
        {
            var useLiveDqn = leftActorMode == ArenaActorControlMode.LiveCnnDqnTraining
                || rightActorMode == ArenaActorControlMode.LiveCnnDqnTraining;
            var liveDqnClient = GetComponent<ArenaLiveDqnTrainingClient>();
            if (useLiveDqn)
            {
                liveDqnClient ??= gameObject.AddComponent<ArenaLiveDqnTrainingClient>();
                liveDqnClient.enabled = true;
            }
            else if (liveDqnClient != null)
            {
                liveDqnClient.enabled = false;
            }
        }

        private void ConfigureActorControl(ArenaCharacterController actor, ArenaActorControlMode mode)
        {
            if (actor == null)
            {
                return;
            }

            var useAi = mode != ArenaActorControlMode.Human;
            actor.IsGhost = useAi;

            var ghostController = actor.GetComponent<ArenaGhostController>();
            var onnxPolicy = actor.GetComponent<ArenaGhostOnnxPolicy>();
            var rasterOnnxPolicy = actor.GetComponent<ArenaRasterOnnxPolicy>();
            var mlAgent = actor.GetComponent<ArenaMlAgent>();
            if (!useAi)
            {
                if (ghostController != null)
                {
                    ghostController.enabled = true;
                    ghostController.SetControlActive(false);
                }

                if (onnxPolicy != null)
                {
                    onnxPolicy.enabled = false;
                }

                if (rasterOnnxPolicy != null)
                {
                    rasterOnnxPolicy.SetAutoStep(false);
                    rasterOnnxPolicy.enabled = false;
                }

                if (mlAgent != null)
                {
                    mlAgent.SetControlActive(false);
                }

                actor.DebugRouteName = "Human";
                return;
            }

            ghostController ??= actor.gameObject.AddComponent<ArenaGhostController>();
            if (mode == ArenaActorControlMode.LiveCnnDqnTraining)
            {
                ghostController.enabled = true;
                ghostController.SetControlActive(false);
                if (onnxPolicy != null)
                {
                    onnxPolicy.enabled = false;
                }

                if (rasterOnnxPolicy != null)
                {
                    rasterOnnxPolicy.SetAutoStep(false);
                    rasterOnnxPolicy.enabled = false;
                }

                if (mlAgent != null)
                {
                    mlAgent.SetControlActive(false);
                }

                actor.DebugRouteName = "LiveCNN-DQN";
                return;
            }

            if (mode == ArenaActorControlMode.MlAgents)
            {
                mlAgent ??= actor.gameObject.AddComponent<ArenaMlAgent>();
                mlAgent.SetControlActive(true);
                ghostController.enabled = true;
                ghostController.SetControlActive(false);
                if (onnxPolicy != null)
                {
                    onnxPolicy.enabled = false;
                }

                if (rasterOnnxPolicy != null)
                {
                    rasterOnnxPolicy.SetAutoStep(false);
                    rasterOnnxPolicy.enabled = false;
                }

                actor.DebugRouteName = "MLAgents";
                return;
            }

            if (mlAgent != null)
            {
                mlAgent.SetControlActive(false);
            }

            var useVectorOnnxPolicy = mode == ArenaActorControlMode.OnnxInference
                || mode == ArenaActorControlMode.DqnInference;
            var useRasterOnnxPolicy = mode == ArenaActorControlMode.RasterDqnInference;

            if (useVectorOnnxPolicy)
            {
                onnxPolicy ??= actor.gameObject.AddComponent<ArenaGhostOnnxPolicy>();
                onnxPolicy.enabled = true;
            }
            else if (onnxPolicy != null)
            {
                onnxPolicy.enabled = false;
            }

            if (useRasterOnnxPolicy)
            {
                rasterOnnxPolicy ??= actor.gameObject.AddComponent<ArenaRasterOnnxPolicy>();
                rasterOnnxPolicy.enabled = true;
                rasterOnnxPolicy.SetAutoStep(false);
            }
            else if (rasterOnnxPolicy != null)
            {
                rasterOnnxPolicy.SetAutoStep(false);
                rasterOnnxPolicy.enabled = false;
            }

            ghostController.enabled = true;
            ghostController.SetControlActive(true);
            ghostController.SetPolicyMode(ToGhostPolicyMode(mode));
        }

        private static ArenaGhostPolicyMode ToGhostPolicyMode(ArenaActorControlMode mode)
        {
            return mode switch
            {
                ArenaActorControlMode.OnnxInference => ArenaGhostPolicyMode.OnnxInference,
                ArenaActorControlMode.DqnInference => ArenaGhostPolicyMode.DqnInference,
                ArenaActorControlMode.RasterDqnInference => ArenaGhostPolicyMode.RasterDqnInference,
                _ => ArenaGhostPolicyMode.RuleBased
            };
        }

        public Vector3 GetSpawnPoint(ArenaSide side)
        {
            return side == ArenaSide.Left ? new Vector3(-8.8f, 0f, 0f) : new Vector3(8.8f, 0f, 0f);
        }

        public Vector3 GetSharedBasePoint()
        {
            return sharedBase != null ? sharedBase.transform.position : new Vector3(0f, -4.45f, 0f);
        }

        public void Deliver(ArenaCharacterController controller)
        {
            if (roundTransition || !controller.HasItem)
            {
                return;
            }

            LastScoringSide = controller.Side;

            if (controller.Side == ArenaSide.Left)
            {
                leftScore++;
            }
            else
            {
                rightScore++;
            }

            controller.DropItem();
            ReportReward(ArenaRewardEventType.ItemDelivered, controller.Side, 5f, "item_delivered");
            EndRound($"{controller.Side}_scored");
        }

        public void BeginRound()
        {
            roundTransition = false;
            roundIndex++;
            roundStartTime = Time.time;
            pendingLeftDqnReward = 0f;
            pendingRightDqnReward = 0f;
            pendingLeftMlAgentReward = 0f;
            pendingRightMlAgentReward = 0f;
            pendingLeftLiveDqnReward = 0f;
            pendingRightLiveDqnReward = 0f;
            pendingLeftCausalResult = default;
            pendingRightCausalResult = default;
            hasPreviousPlayerDqnStep = false;
            ResetLogSamplingTimers();
            ResetRoundStats();
            ResetEnvironmentObjects();
            PrepareRoundLogFile();

            player.ResetActor(GetSpawnPoint(ArenaSide.Left));
            ghost.ResetActor(GetSpawnPoint(ArenaSide.Right));
            currentItemSpawnIndex = ResolveItemSpawnIndex();
            item.ResetToSpawn(itemSpawnPoints[currentItemSpawnIndex]);
            ApplyReturnToBaseCurriculumIfNeeded();
            var rasterEncoder = GetComponent<ArenaRasterEncoder>();
            if (rasterEncoder != null)
            {
                rasterEncoder.ResetFrameStack();
            }

            ReportReward(ArenaRewardEventType.RoundStart, ArenaSide.Left, 0f, "round_start");
            ReportReward(ArenaRewardEventType.RoundStart, ArenaSide.Right, 0f, "round_start");
        }

        private void ApplyReturnToBaseCurriculumIfNeeded()
        {
            if (!ShouldUseReturnToBaseCurriculum())
            {
                return;
            }

            var holderSide = ResolveReturnToBaseHolderSide();
            var holder = holderSide == ArenaSide.Left ? player : ghost;
            if (holder == null || item == null)
            {
                return;
            }

            var start = holderSide == ArenaSide.Left ? returnToBaseLeftStart : returnToBaseRightStart;
            holder.ResetActor(new Vector3(start.x, start.y, 0f));
            item.ResetToSpawn(holder.transform.position);
            holder.PickupItem(item, reportReward: false);
        }

        private bool ShouldUseReturnToBaseCurriculum()
        {
            if (!enableReturnToBaseCurriculum)
            {
                return false;
            }

            if (returnToBaseCurriculumUntilRound > 0 && roundIndex > returnToBaseCurriculumUntilRound)
            {
                return false;
            }

            return Random.value <= returnToBaseCurriculumChance;
        }

        private ArenaSide ResolveReturnToBaseHolderSide()
        {
            if (alternateReturnToBaseHolder)
            {
                return roundIndex % 2 == 0 ? ArenaSide.Right : ArenaSide.Left;
            }

            return Random.value < 0.5f ? ArenaSide.Left : ArenaSide.Right;
        }

        private int ResolveItemSpawnIndex()
        {
            if (itemSpawnPoints == null || itemSpawnPoints.Length == 0)
            {
                return 0;
            }

            switch (itemSpawnMode)
            {
                case ArenaItemSpawnMode.Cycle:
                {
                    var spawnIndex = Mathf.Clamp(nextCycleItemSpawnIndex, 0, itemSpawnPoints.Length - 1);
                    nextCycleItemSpawnIndex = (spawnIndex + 1) % itemSpawnPoints.Length;
                    return spawnIndex;
                }
                case ArenaItemSpawnMode.Random:
                    return Random.Range(0, itemSpawnPoints.Length);
                case ArenaItemSpawnMode.Fixed:
                default:
                    return Mathf.Clamp(fixedItemSpawnIndex, 0, itemSpawnPoints.Length - 1);
            }
        }

        private void ResetEnvironmentObjects()
        {
            roundDoorOpenCount = 0;
            roundSwitchActivationCount = 0;

            if (arenaDoors != null)
            {
                foreach (var arenaDoor in arenaDoors)
                {
                    arenaDoor?.ResetDoor();
                }
            }

            if (arenaSwitches != null)
            {
                foreach (var arenaSwitch in arenaSwitches)
                {
                    arenaSwitch?.ResetSwitch();
                }
            }

            if (movingObstacles != null)
            {
                foreach (var movingObstacle in movingObstacles)
                {
                    movingObstacle?.ResetMotion();
                }
            }
        }

        private void Update()
        {
            if (!HasRequiredReferences())
            {
                return;
            }

            if (!roundTransition
                && roundDurationSeconds > 0f
                && Time.time - roundStartTime >= roundDurationSeconds)
            {
                ReportReward(
                    ArenaRewardEventType.RoundTimeout,
                    ArenaSide.Left,
                    GetRoundTimeoutReward(player),
                    GetRoundTimeoutNote(player));
                ReportReward(
                    ArenaRewardEventType.RoundTimeout,
                    ArenaSide.Right,
                    GetRoundTimeoutReward(ghost),
                    GetRoundTimeoutNote(ghost));
                EndRound(ResolveTimeoutOutcome());
            }

            var leftObservation = BuildObservation(ArenaSide.Left);
            var leftAction = player.GetCurrentActionSnapshot();
            AccumulateFrameStats(ArenaSide.Left, leftObservation, leftAction);
            LogStep(ArenaSide.Left, roundIndex, leftObservation, leftAction);
            LogDqnTransitionIfReady(leftObservation, leftAction);

            var rightObservation = BuildObservation(ArenaSide.Right);
            var rightAction = ghost.GetCurrentActionSnapshot();
            AccumulateFrameStats(ArenaSide.Right, rightObservation, rightAction);
        }

        private float GetRoundTimeoutReward(ArenaCharacterController actor)
        {
            return actor != null && !actor.HasItem ? timeoutNoItemPenalty : 0f;
        }

        private static string GetRoundTimeoutNote(ArenaCharacterController actor)
        {
            return actor != null && actor.HasItem ? "round_timeout_has_item" : "round_timeout_no_item";
        }

        public ArenaObservationSnapshot BuildObservation(ArenaSide side)
        {
            EnsureEnvironmentReferencesAvailable();
            var self = side == ArenaSide.Left ? player : ghost;
            var opponent = side == ArenaSide.Left ? ghost : player;
            var basePosition = sharedBase != null ? (Vector2)sharedBase.transform.position : Vector2.zero;
            var targetType = ResolveTargetType(self, opponent);
            var targetPosition = ResolveTargetPosition(targetType, opponent, basePosition);
            var selfPosition = (Vector2)self.transform.position;
            var itemPosition = (Vector2)item.transform.position;
            var opponentPosition = (Vector2)opponent.transform.position;
            var itemDelta = itemPosition - selfPosition;
            var baseDelta = basePosition - selfPosition;
            var opponentDelta = opponentPosition - selfPosition;
            var nearestDoor = FindNearestDoor(selfPosition);
            var nearestSwitch = FindNearestSwitch(selfPosition, nearestDoor) ?? FindNearestSwitch(selfPosition);
            var nearestMovingObstacle = FindNearestMovingObstacle(selfPosition);
            var doorPosition = nearestDoor != null ? (Vector2)nearestDoor.transform.position : selfPosition;
            var switchPosition = nearestSwitch != null ? (Vector2)nearestSwitch.transform.position : selfPosition;
            var movingObstaclePosition = nearestMovingObstacle != null
                ? (Vector2)nearestMovingObstacle.transform.position
                : selfPosition;
            var movingObstacleDelta = movingObstaclePosition - selfPosition;
            var movingObstacleVelocity = nearestMovingObstacle != null ? nearestMovingObstacle.Velocity : Vector2.zero;
            var shortcutBlocked = nearestDoor != null
                && !nearestDoor.IsOpen
                && IsShortcutRelevant(selfPosition, targetPosition, (Vector2)nearestDoor.transform.position);

            return new ArenaObservationSnapshot
            {
                selfPosition = selfPosition,
                opponentPosition = opponentPosition,
                itemPosition = itemPosition,
                basePosition = basePosition,
                selfHasItem = self.HasItem,
                opponentHasItem = opponent.HasItem,
                itemDelta = itemDelta,
                baseDelta = baseDelta,
                opponentDelta = opponentDelta,
                distanceToItem = itemDelta.magnitude,
                distanceToBase = baseDelta.magnitude,
                distanceToOpponent = opponentDelta.magnitude,
                distanceToTarget = Vector2.Distance(selfPosition, targetPosition),
                targetPosition = targetPosition,
                targetType = targetType,
                wallAhead = self.IsWallAhead(),
                roundElapsedTime = Time.time - roundStartTime,
                doorDelta = doorPosition - selfPosition,
                doorOpen = nearestDoor == null || nearestDoor.IsOpen,
                switchDelta = switchPosition - selfPosition,
                switchActive = nearestSwitch != null && nearestSwitch.IsActive,
                movingObstacleDelta = movingObstacleDelta,
                movingObstacleVelocity = movingObstacleVelocity,
                movingObstacleAhead = IsMovingObstacleAhead(selfPosition, targetPosition, movingObstacleDelta),
                shortcutBlocked = shortcutBlocked,
                detourNeeded = shortcutBlocked
            };
        }

        private static bool IsShortcutRelevant(Vector2 selfPosition, Vector2 targetPosition, Vector2 doorPosition)
        {
            var targetDelta = targetPosition - selfPosition;
            var doorDelta = doorPosition - selfPosition;
            if (targetDelta.sqrMagnitude < 0.001f || doorDelta.sqrMagnitude > 12.25f)
            {
                return false;
            }

            return Vector2.Dot(targetDelta.normalized, doorDelta.normalized) > 0.35f;
        }

        private ArenaDoor FindNearestDoor(Vector2 position)
        {
            ArenaDoor nearest = null;
            var nearestDistance = float.MaxValue;
            if (arenaDoors == null)
            {
                return null;
            }

            foreach (var arenaDoor in arenaDoors)
            {
                if (arenaDoor == null)
                {
                    continue;
                }

                var distance = Vector2.SqrMagnitude((Vector2)arenaDoor.transform.position - position);
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearest = arenaDoor;
                }
            }

            return nearest;
        }

        private ArenaSwitch FindNearestSwitch(Vector2 position, ArenaDoor linkedDoor = null)
        {
            ArenaSwitch nearest = null;
            var nearestDistance = float.MaxValue;
            if (arenaSwitches == null)
            {
                return null;
            }

            foreach (var arenaSwitch in arenaSwitches)
            {
                if (arenaSwitch == null)
                {
                    continue;
                }

                if (linkedDoor != null && arenaSwitch.LinkedDoor != linkedDoor)
                {
                    continue;
                }

                var distance = Vector2.SqrMagnitude((Vector2)arenaSwitch.transform.position - position);
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearest = arenaSwitch;
                }
            }

            return nearest;
        }

        private ArenaMovingObstacle FindNearestMovingObstacle(Vector2 position)
        {
            ArenaMovingObstacle nearest = null;
            var nearestDistance = float.MaxValue;
            if (movingObstacles == null)
            {
                return null;
            }

            foreach (var movingObstacle in movingObstacles)
            {
                if (movingObstacle == null)
                {
                    continue;
                }

                var distance = Vector2.SqrMagnitude((Vector2)movingObstacle.transform.position - position);
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearest = movingObstacle;
                }
            }

            return nearest;
        }

        private static bool IsMovingObstacleAhead(Vector2 selfPosition, Vector2 targetPosition, Vector2 obstacleDelta)
        {
            if (obstacleDelta.sqrMagnitude > 4f)
            {
                return false;
            }

            var targetDelta = targetPosition - selfPosition;
            if (targetDelta.sqrMagnitude < 0.001f)
            {
                return false;
            }

            return Vector2.Dot(targetDelta.normalized, obstacleDelta.normalized) > 0.55f;
        }

        private ArenaTargetType ResolveTargetType(ArenaCharacterController self, ArenaCharacterController opponent)
        {
            if (self.HasItem)
            {
                return ArenaTargetType.Base;
            }

            if (opponent.HasItem)
            {
                return ArenaTargetType.Opponent;
            }

            return ArenaTargetType.Item;
        }

        private Vector2 ResolveTargetPosition(ArenaTargetType targetType, ArenaCharacterController opponent, Vector2 basePosition)
        {
            return targetType switch
            {
                ArenaTargetType.Base => basePosition,
                ArenaTargetType.Opponent => opponent != null ? (Vector2)opponent.transform.position : Vector2.zero,
                _ => item != null ? (Vector2)item.transform.position : Vector2.zero,
            };
        }

        public void ReportReward(ArenaRewardEventType eventType, ArenaSide actorSide, float rewardDelta, string note)
        {
            ApplyRewardToRoundState(eventType, actorSide, rewardDelta, note);

            AddPendingDqnReward(actorSide, rewardDelta);
            AddPendingMlAgentReward(actorSide, rewardDelta);
            AddPendingLiveDqnReward(actorSide, rewardDelta);

            if (!writeLogsToFile)
            {
                return;
            }

            EnsureRoundLogFileReady();

            var rewardEvent = new ArenaRewardEvent
            {
                eventType = eventType,
                actorSide = actorSide,
                rewardDelta = rewardDelta,
                timestamp = Time.time - roundStartTime,
                roundIndex = roundIndex,
                note = note
            };

            var json = JsonUtility.ToJson(rewardEvent);
            File.AppendAllText(currentRoundOutputPath, json + "\n", Encoding.UTF8);
        }

        public float ConsumeMlAgentReward(ArenaSide side)
        {
            if (side == ArenaSide.Left)
            {
                var reward = pendingLeftMlAgentReward;
                pendingLeftMlAgentReward = 0f;
                return reward;
            }

            var rightReward = pendingRightMlAgentReward;
            pendingRightMlAgentReward = 0f;
            return rightReward;
        }

        public float ConsumeLiveDqnReward(ArenaSide side)
        {
            if (side == ArenaSide.Left)
            {
                var reward = pendingLeftLiveDqnReward;
                pendingLeftLiveDqnReward = 0f;
                return reward;
            }

            var rightReward = pendingRightLiveDqnReward;
            pendingRightLiveDqnReward = 0f;
            return rightReward;
        }

        public void ReportDoorOpened(ArenaDoor arenaDoor, ArenaSide? openedBy)
        {
            if (arenaDoor == null || roundTransition)
            {
                return;
            }

            roundDoorOpenCount++;
            if (!openedBy.HasValue)
            {
                return;
            }

            ref var stats = ref GetMutableStats(openedBy.Value);
            stats.doorOpenCount++;
            AddPendingMlAgentReward(openedBy.Value, mlDoorOpenReward);
            AddPendingDqnReward(openedBy.Value, mlDoorOpenReward);
            AddPendingLiveDqnReward(openedBy.Value, mlDoorOpenReward);
        }

        public void ReportSwitchActivated(ArenaSwitch arenaSwitch, ArenaSide activatedBy, bool linkedDoorWasAlreadyOpen)
        {
            if (arenaSwitch == null || roundTransition)
            {
                return;
            }

            roundSwitchActivationCount++;
            ref var stats = ref GetMutableStats(activatedBy);
            stats.switchActivationCount++;

            if (linkedDoorWasAlreadyOpen && openDoorSwitchPenalty < 0f)
            {
                ReportReward(
                    ArenaRewardEventType.OpenDoorSwitchPressed,
                    activatedBy,
                    openDoorSwitchPenalty,
                    "switch_pressed_while_door_open");
            }
        }

        public void ReportOpenDoorSwitchHeld(ArenaSwitch arenaSwitch, ArenaSide activatedBy)
        {
            if (arenaSwitch == null || roundTransition || openDoorSwitchPenalty >= 0f)
            {
                return;
            }

            ReportReward(
                ArenaRewardEventType.OpenDoorSwitchPressed,
                activatedBy,
                openDoorSwitchPenalty,
                "switch_held_while_door_open");
        }

        public void ReportShoveAttempt(ArenaSide actorSide, bool succeeded, bool forcedItemDrop)
        {
            ref var stats = ref GetMutableStats(actorSide);
            stats.shoveAttemptCount++;

            ref var result = ref GetMutableCausalResult(actorSide);
            result.shoveAttempted = true;

            if (!succeeded)
            {
                return;
            }

            stats.shoveSuccessCount++;
            result.shoveSucceeded = true;

            if (forcedItemDrop)
            {
                stats.forcedItemDropCount++;
                result.forcedItemDrop = true;
            }
        }

        private void LogStep(
            ArenaSide actorSide,
            int currentRoundIndex,
            ArenaObservationSnapshot observation,
            ArenaActionSnapshot action)
        {
            if (!writeLogsToFile)
            {
                return;
            }

            if (!ShouldWriteStepLog(actorSide))
            {
                return;
            }

            EnsureRoundLogFileReady();

            var json = JsonUtility.ToJson(new ArenaStepLog
            {
                logType = "Step",
                actorSide = actorSide.ToString(),
                roundIndex = currentRoundIndex,
                observation = observation,
                action = action,
                timestamp = Time.time - roundStartTime
            });

            File.AppendAllText(currentRoundOutputPath, json + "\n", Encoding.UTF8);

            var causalResult = ConsumePendingCausalResult(actorSide);
            var causalJson = JsonUtility.ToJson(new ArenaCausalStepLog
            {
                logType = "CausalStep",
                actorSide = actorSide.ToString(),
                roundIndex = currentRoundIndex,
                timestamp = Time.time - roundStartTime,
                observation = observation,
                action = action,
                result = causalResult,
                causalTags = BuildCausalTags(observation, action)
            });

            File.AppendAllText(currentRoundOutputPath, causalJson + "\n", Encoding.UTF8);
        }

        private void LogDqnTransitionIfReady(ArenaObservationSnapshot currentObservation, ArenaActionSnapshot currentAction)
        {
            if (!writeLogsToFile || !writeDqnTransitions)
            {
                StorePreviousPlayerDqnStep(currentObservation, currentAction);
                return;
            }

            if (hasPreviousPlayerDqnStep)
            {
                var done = roundTransition;
                if (!done && !ShouldWriteDqnTransition())
                {
                    return;
                }

                var reward = ConsumePendingDqnReward(ArenaSide.Left)
                    + ComputeShapingReward(previousPlayerObservation, currentObservation, previousPlayerAction);
                var transition = new ArenaDqnTransitionLog
                {
                    transitionType = "DqnTransition",
                    actorSide = ArenaSide.Left.ToString(),
                    roundIndex = roundIndex,
                    timestamp = Time.time - roundStartTime,
                    state = previousPlayerObservation,
                    action = ToDqnAction(previousPlayerAction),
                    reward = reward,
                    nextState = currentObservation,
                    done = done
                };

                var json = JsonUtility.ToJson(transition);
                File.AppendAllText(currentRoundOutputPath, json + "\n", Encoding.UTF8);

                if (done)
                {
                    hasPreviousPlayerDqnStep = false;
                    return;
                }
            }

            StorePreviousPlayerDqnStep(currentObservation, currentAction);
        }

        private void ResetLogSamplingTimers()
        {
            var elapsed = Time.time - roundStartTime;
            nextLeftStepLogTime = elapsed;
            nextRightStepLogTime = elapsed;
            nextPlayerDqnTransitionLogTime = elapsed + Mathf.Max(0f, dqnTransitionLogIntervalSeconds);
        }

        private bool ShouldWriteStepLog(ArenaSide side)
        {
            var interval = Mathf.Max(0f, stepLogIntervalSeconds);
            if (interval <= 0f)
            {
                return true;
            }

            var elapsed = Time.time - roundStartTime;
            var nextTime = side == ArenaSide.Left ? nextLeftStepLogTime : nextRightStepLogTime;
            if (elapsed + Mathf.Epsilon < nextTime)
            {
                return false;
            }

            if (side == ArenaSide.Left)
            {
                nextLeftStepLogTime = elapsed + interval;
            }
            else
            {
                nextRightStepLogTime = elapsed + interval;
            }

            return true;
        }

        private bool ShouldWriteDqnTransition()
        {
            var interval = Mathf.Max(0f, dqnTransitionLogIntervalSeconds);
            if (interval <= 0f)
            {
                return true;
            }

            var elapsed = Time.time - roundStartTime;
            if (elapsed + Mathf.Epsilon < nextPlayerDqnTransitionLogTime)
            {
                return false;
            }

            nextPlayerDqnTransitionLogTime = elapsed + interval;
            return true;
        }

        private void StorePreviousPlayerDqnStep(ArenaObservationSnapshot observation, ArenaActionSnapshot action)
        {
            previousPlayerObservation = observation;
            previousPlayerAction = action;
            hasPreviousPlayerDqnStep = true;
        }

        private void AddPendingDqnReward(ArenaSide side, float rewardDelta)
        {
            if (side == ArenaSide.Left)
            {
                pendingLeftDqnReward += rewardDelta;
            }
            else
            {
                pendingRightDqnReward += rewardDelta;
            }
        }

        private void AddPendingMlAgentReward(ArenaSide side, float rewardDelta)
        {
            if (side == ArenaSide.Left)
            {
                pendingLeftMlAgentReward += rewardDelta;
                return;
            }

            pendingRightMlAgentReward += rewardDelta;
        }

        private void AddPendingLiveDqnReward(ArenaSide side, float rewardDelta)
        {
            if (side == ArenaSide.Left)
            {
                pendingLeftLiveDqnReward += rewardDelta;
                return;
            }

            pendingRightLiveDqnReward += rewardDelta;
        }

        private float ConsumePendingDqnReward(ArenaSide side)
        {
            if (side == ArenaSide.Left)
            {
                var reward = pendingLeftDqnReward;
                pendingLeftDqnReward = 0f;
                return reward;
            }

            var rightReward = pendingRightDqnReward;
            pendingRightDqnReward = 0f;
            return rightReward;
        }

        private float ComputeShapingReward(
            ArenaObservationSnapshot previousObservation,
            ArenaObservationSnapshot currentObservation,
            ArenaActionSnapshot previousAction)
        {
            var progressReward = ComputeEuclideanProgressReward(previousObservation, currentObservation);
            var flowDirectionReward = 0f;
            if (TryComputePathGuidanceReward(
                    previousObservation,
                    currentObservation,
                    previousAction,
                    out var pathProgressReward,
                    out flowDirectionReward)
                && usePathDistanceReward)
            {
                progressReward = pathProgressReward;
            }

            var wallPenalty = previousObservation.wallAhead
                && previousAction.moveAction != ArenaMoveAction.Idle
                ? dqnWallActionPenalty
                : 0f;
            var invalidShovePenalty = previousAction.shovePressed
                && !HasUsefulShoveTarget(previousObservation)
                ? dqnInvalidShovePenalty
                : 0f;
            var closedDoorPenalty = previousObservation.shortcutBlocked
                && previousAction.moveAction != ArenaMoveAction.Idle
                ? mlClosedDoorActionPenalty
                : 0f;
            var movingObstaclePenalty = previousObservation.movingObstacleAhead
                && previousAction.moveAction != ArenaMoveAction.Idle
                ? mlMovingObstacleActionPenalty
                : 0f;

            return dqnStepPenalty
                + progressReward
                + flowDirectionReward
                + wallPenalty
                + invalidShovePenalty
                + closedDoorPenalty
                + movingObstaclePenalty;
        }

        private float ComputeEuclideanProgressReward(
            ArenaObservationSnapshot previousObservation,
            ArenaObservationSnapshot currentObservation)
        {
            var previousDistance = Vector2.Distance(previousObservation.selfPosition, previousObservation.targetPosition);
            var currentDistance = Vector2.Distance(currentObservation.selfPosition, currentObservation.targetPosition);
            var progressScale = ResolveProgressRewardScale(previousObservation.targetType);
            return Mathf.Clamp(
                (previousDistance - currentDistance) * progressScale,
                -dqnProgressRewardClamp,
                dqnProgressRewardClamp);
        }

        private bool TryComputePathGuidanceReward(
            ArenaObservationSnapshot previousObservation,
            ArenaObservationSnapshot currentObservation,
            ArenaActionSnapshot previousAction,
            out float pathProgressReward,
            out float flowDirectionReward)
        {
            pathProgressReward = 0f;
            flowDirectionReward = 0f;
            if (!usePathDistanceReward && !useFlowFieldDirectionReward)
            {
                return false;
            }

            if (!TryBuildPathDistanceField(previousObservation.targetPosition, out var distances, out var blocked))
            {
                return false;
            }

            var previousIndex = FindNearestReachablePathIndex(WorldToPathCell(previousObservation.selfPosition), distances);
            var currentIndex = FindNearestReachablePathIndex(WorldToPathCell(currentObservation.selfPosition), distances);
            if (previousIndex < 0 || currentIndex < 0)
            {
                return false;
            }

            if (usePathDistanceReward)
            {
                var pathProgress = distances[previousIndex] - distances[currentIndex];
                pathProgressReward = Mathf.Clamp(
                    pathProgress * pathDistanceRewardScale,
                    -dqnProgressRewardClamp,
                    dqnProgressRewardClamp);
            }

            if (useFlowFieldDirectionReward)
            {
                flowDirectionReward = ComputeFlowFieldDirectionReward(previousIndex, previousAction.moveAction, distances, blocked);
            }

            return true;
        }

        private float ComputeFlowFieldDirectionReward(
            int previousIndex,
            ArenaMoveAction moveAction,
            int[] distances,
            bool[] blocked)
        {
            if (moveAction == ArenaMoveAction.Idle || previousIndex < 0 || previousIndex >= distances.Length)
            {
                return 0f;
            }

            var currentDistance = distances[previousIndex];
            if (currentDistance <= 0)
            {
                return 0f;
            }

            var cell = PathIndexToCell(previousIndex);
            var delta = MoveActionToPathCellDelta(moveAction);
            var nextX = cell.x + delta.x;
            var nextY = cell.y + delta.y;
            if (!IsPathCellInside(nextY, nextX))
            {
                return pathFlowDirectionPenalty;
            }

            var nextIndex = PathIndex(nextY, nextX);
            if (blocked[nextIndex] || distances[nextIndex] < 0)
            {
                return pathFlowDirectionPenalty;
            }

            if (distances[nextIndex] < currentDistance)
            {
                return pathFlowDirectionReward;
            }

            if (distances[nextIndex] > currentDistance)
            {
                return pathFlowDirectionPenalty;
            }

            return 0f;
        }

        private bool TryBuildPathDistanceField(Vector2 targetPosition, out int[] distances, out bool[] blocked)
        {
            var cellCount = PathGridWidth * PathGridHeight;
            distances = new int[cellCount];
            blocked = new bool[cellCount];
            for (var i = 0; i < distances.Length; i++)
            {
                distances[i] = -1;
            }

            BuildPathBlockedCells(blocked);
            var targetIndex = FindNearestWalkablePathIndex(WorldToPathCell(targetPosition), blocked);
            if (targetIndex < 0)
            {
                return false;
            }

            var queue = new int[cellCount];
            var head = 0;
            var tail = 0;
            queue[tail++] = targetIndex;
            distances[targetIndex] = 0;
            while (head < tail)
            {
                var index = queue[head++];
                var cell = PathIndexToCell(index);
                var nextDistance = distances[index] + 1;
                TryVisitPathNeighbor(cell.x + 1, cell.y, nextDistance, distances, blocked, queue, ref tail);
                TryVisitPathNeighbor(cell.x - 1, cell.y, nextDistance, distances, blocked, queue, ref tail);
                TryVisitPathNeighbor(cell.x, cell.y + 1, nextDistance, distances, blocked, queue, ref tail);
                TryVisitPathNeighbor(cell.x, cell.y - 1, nextDistance, distances, blocked, queue, ref tail);
            }

            return true;
        }

        private void TryVisitPathNeighbor(
            int x,
            int y,
            int distance,
            int[] distances,
            bool[] blocked,
            int[] queue,
            ref int tail)
        {
            if (!IsPathCellInside(y, x))
            {
                return;
            }

            var index = PathIndex(y, x);
            if (blocked[index] || distances[index] >= 0)
            {
                return;
            }

            distances[index] = distance;
            queue[tail++] = index;
        }

        private void BuildPathBlockedCells(bool[] blocked)
        {
            var root = transform.parent != null ? transform.parent : transform;
            foreach (var collider in root.GetComponentsInChildren<BoxCollider2D>(includeInactive: true))
            {
                if (ShouldIgnorePathCollider(collider))
                {
                    continue;
                }

                MarkPathColliderBounds(blocked, collider.bounds);
            }
        }

        private bool ShouldIgnorePathCollider(BoxCollider2D collider)
        {
            if (collider == null || collider.isTrigger)
            {
                return true;
            }

            if (collider.GetComponentInParent<ArenaCharacterController>() != null
                || collider.GetComponentInParent<ArenaItem>() != null
                || collider.GetComponentInParent<ArenaBaseZone>() != null
                || collider.GetComponentInParent<ArenaSwitch>() != null)
            {
                return true;
            }

            var door = collider.GetComponentInParent<ArenaDoor>();
            if (door != null)
            {
                return !pathTreatClosedDoorsAsBlocked || door.IsOpen;
            }

            var movingObstacle = collider.GetComponentInParent<ArenaMovingObstacle>();
            if (movingObstacle != null)
            {
                return !pathTreatMovingObstaclesAsBlocked;
            }

            return false;
        }

        private void MarkPathColliderBounds(bool[] blocked, Bounds bounds)
        {
            var min = WorldToPathCell(bounds.min);
            var max = WorldToPathCell(bounds.max);
            var minX = Mathf.Min(min.x, max.x);
            var maxX = Mathf.Max(min.x, max.x);
            var minY = Mathf.Min(min.y, max.y);
            var maxY = Mathf.Max(min.y, max.y);
            for (var y = minY; y <= maxY; y++)
            {
                for (var x = minX; x <= maxX; x++)
                {
                    if (IsPathCellInside(y, x))
                    {
                        blocked[PathIndex(y, x)] = true;
                    }
                }
            }
        }

        private int FindNearestWalkablePathIndex(Vector2Int origin, bool[] blocked)
        {
            var maxRadius = Mathf.Max(PathGridWidth, PathGridHeight);
            for (var radius = 0; radius <= maxRadius; radius++)
            {
                for (var y = origin.y - radius; y <= origin.y + radius; y++)
                {
                    for (var x = origin.x - radius; x <= origin.x + radius; x++)
                    {
                        if (Mathf.Abs(x - origin.x) + Mathf.Abs(y - origin.y) != radius)
                        {
                            continue;
                        }

                        if (!IsPathCellInside(y, x))
                        {
                            continue;
                        }

                        var index = PathIndex(y, x);
                        if (!blocked[index])
                        {
                            return index;
                        }
                    }
                }
            }

            return -1;
        }

        private int FindNearestReachablePathIndex(Vector2Int origin, int[] distances)
        {
            var maxRadius = Mathf.Max(PathGridWidth, PathGridHeight);
            for (var radius = 0; radius <= maxRadius; radius++)
            {
                for (var y = origin.y - radius; y <= origin.y + radius; y++)
                {
                    for (var x = origin.x - radius; x <= origin.x + radius; x++)
                    {
                        if (Mathf.Abs(x - origin.x) + Mathf.Abs(y - origin.y) != radius)
                        {
                            continue;
                        }

                        if (!IsPathCellInside(y, x))
                        {
                            continue;
                        }

                        var index = PathIndex(y, x);
                        if (distances[index] >= 0)
                        {
                            return index;
                        }
                    }
                }
            }

            return -1;
        }

        private Vector2Int WorldToPathCell(Vector2 position)
        {
            var normalized = new Vector2(
                Mathf.InverseLerp(pathWorldMin.x, pathWorldMax.x, position.x),
                Mathf.InverseLerp(pathWorldMin.y, pathWorldMax.y, position.y));
            var x = Mathf.Clamp(Mathf.FloorToInt(normalized.x * PathGridWidth), 0, PathGridWidth - 1);
            var y = Mathf.Clamp(Mathf.FloorToInt((1f - normalized.y) * PathGridHeight), 0, PathGridHeight - 1);
            return new Vector2Int(x, y);
        }

        private Vector2Int PathIndexToCell(int index)
        {
            return new Vector2Int(index % PathGridWidth, index / PathGridWidth);
        }

        private Vector2Int MoveActionToPathCellDelta(ArenaMoveAction action)
        {
            return action switch
            {
                ArenaMoveAction.MoveUp => new Vector2Int(0, -1),
                ArenaMoveAction.MoveDown => new Vector2Int(0, 1),
                ArenaMoveAction.MoveLeft => new Vector2Int(-1, 0),
                ArenaMoveAction.MoveRight => new Vector2Int(1, 0),
                _ => Vector2Int.zero
            };
        }

        private int PathIndex(int y, int x)
        {
            return y * PathGridWidth + x;
        }

        private bool IsPathCellInside(int y, int x)
        {
            return x >= 0 && y >= 0 && x < PathGridWidth && y < PathGridHeight;
        }

        private int PathGridWidth => Mathf.Max(4, pathGridWidth);
        private int PathGridHeight => Mathf.Max(4, pathGridHeight);

        private float ResolveProgressRewardScale(ArenaTargetType targetType)
        {
            return targetType switch
            {
                ArenaTargetType.Base => dqnBaseTargetProgressRewardScale,
                ArenaTargetType.Opponent => dqnOpponentTargetProgressRewardScale,
                ArenaTargetType.Item => dqnItemTargetProgressRewardScale,
                _ => dqnTargetProgressRewardScale
            };
        }

        private bool HasUsefulShoveTarget(ArenaObservationSnapshot observation)
        {
            return observation.opponentHasItem
                && observation.distanceToOpponent <= Mathf.Max(0.01f, dqnShoveOpportunityDistance);
        }

        public float ComputeSharedAgentShapingReward(
            ArenaObservationSnapshot previousObservation,
            ArenaObservationSnapshot currentObservation,
            ArenaActionSnapshot previousAction)
        {
            return ComputeShapingReward(previousObservation, currentObservation, previousAction);
        }

        public float ComputeSharedAgentShapingReward(
            ArenaObservationSnapshot previousObservation,
            ArenaObservationSnapshot currentObservation,
            ArenaDqnAction previousAction)
        {
            var actionSnapshot = new ArenaActionSnapshot
            {
                moveAction = ToMoveAction(previousAction),
                shovePressed = previousAction == ArenaDqnAction.Shove,
                routeName = "SharedAgent"
            };
            return ComputeShapingReward(previousObservation, currentObservation, actionSnapshot);
        }

        public float ComputeLiveDqnShapingReward(
            ArenaObservationSnapshot previousObservation,
            ArenaObservationSnapshot currentObservation,
            ArenaDqnAction previousAction)
        {
            return ComputeSharedAgentShapingReward(previousObservation, currentObservation, previousAction);
        }

        private static ArenaMoveAction ToMoveAction(ArenaDqnAction action)
        {
            return action switch
            {
                ArenaDqnAction.MoveUp => ArenaMoveAction.MoveUp,
                ArenaDqnAction.MoveDown => ArenaMoveAction.MoveDown,
                ArenaDqnAction.MoveLeft => ArenaMoveAction.MoveLeft,
                ArenaDqnAction.MoveRight => ArenaMoveAction.MoveRight,
                _ => ArenaMoveAction.Idle
            };
        }

        private void ApplyRewardToRoundState(ArenaRewardEventType eventType, ArenaSide actorSide, float rewardDelta, string note)
        {
            ref var stats = ref GetMutableStats(actorSide);
            ref var result = ref GetMutableCausalResult(actorSide);
            result.reward += rewardDelta;

            switch (eventType)
            {
                case ArenaRewardEventType.ItemCollected:
                    stats.itemCollectedCount++;
                    result.pickedItem = true;
                    break;
                case ArenaRewardEventType.ItemDelivered:
                    stats.itemDeliveredCount++;
                    result.scored = true;
                    result.done = true;
                    break;
                case ArenaRewardEventType.ItemLost:
                    if (rewardDelta < 0f)
                    {
                        result.droppedItem = true;
                    }
                    else if (note == "forced_drop")
                    {
                        result.forcedItemDrop = true;
                    }

                    break;
                case ArenaRewardEventType.RoundTimeout:
                    result.done = true;
                    break;
            }
        }

        private void AccumulateFrameStats(
            ArenaSide actorSide,
            ArenaObservationSnapshot observation,
            ArenaActionSnapshot action)
        {
            ref var stats = ref GetMutableStats(actorSide);
            if (action.moveAction == ArenaMoveAction.Idle)
            {
                stats.idleTime += Time.deltaTime;
            }

            if (observation.wallAhead && action.moveAction != ArenaMoveAction.Idle)
            {
                stats.wallBlockedTime += Time.deltaTime;
            }

            if (observation.shortcutBlocked && action.moveAction != ArenaMoveAction.Idle)
            {
                stats.shortcutBlockedCount++;
            }

            if (observation.movingObstacleAhead && action.moveAction != ArenaMoveAction.Idle)
            {
                stats.movingObstacleBlockedTime += Time.deltaTime;
            }
        }

        private void ResetRoundStats()
        {
            leftRoundStats = new ArenaRoundActorStats
            {
                actorSide = ArenaSide.Left.ToString()
            };
            rightRoundStats = new ArenaRoundActorStats
            {
                actorSide = ArenaSide.Right.ToString()
            };
        }

        private void EndRound(string outcome)
        {
            if (roundTransition)
            {
                return;
            }

            roundTransition = true;
            pendingLeftCausalResult.done = true;
            pendingRightCausalResult.done = true;
            WriteRoundSummary(outcome);
            StartCoroutine(ResetRoundAfterDelay());
        }

        private string ResolveTimeoutOutcome()
        {
            if (leftScore > rightScore)
            {
                return "timeout_left_leading";
            }

            if (rightScore > leftScore)
            {
                return "timeout_right_leading";
            }

            return "timeout_draw";
        }

        private void WriteRoundSummary(string outcome)
        {
            if (!writeLogsToFile)
            {
                return;
            }

            EnsureEvaluationSummaryPathReady();
            leftRoundStats.scoreAtEnd = leftScore;
            rightRoundStats.scoreAtEnd = rightScore;

            var summary = new ArenaRoundSummaryLog
            {
                logType = "RoundSummary",
                roundIndex = roundIndex,
                duration = Time.time - roundStartTime,
                outcome = outcome,
                leftScore = leftScore,
                rightScore = rightScore,
                left = leftRoundStats,
                right = rightRoundStats,
                itemSpawnMode = itemSpawnMode.ToString(),
                itemSpawnIndex = currentItemSpawnIndex,
                doorOpenCount = roundDoorOpenCount,
                switchActivationCount = roundSwitchActivationCount
            };

            File.AppendAllText(evaluationSummaryOutputPath, JsonUtility.ToJson(summary) + "\n", Encoding.UTF8);
        }

        private void EnsureEvaluationSummaryPathReady()
        {
            if (string.IsNullOrEmpty(outputDirectoryPath))
            {
                outputDirectoryPath = Path.Combine(Application.persistentDataPath, outputDirectoryName);
            }

            Directory.CreateDirectory(outputDirectoryPath);

            if (string.IsNullOrEmpty(evaluationSummaryOutputPath))
            {
                evaluationSummaryOutputPath = Path.Combine(outputDirectoryPath, evaluationSummaryFileName);
            }
        }

        private ArenaCausalStepResult ConsumePendingCausalResult(ArenaSide side)
        {
            if (side == ArenaSide.Left)
            {
                var result = pendingLeftCausalResult;
                pendingLeftCausalResult = default;
                return result;
            }

            var rightResult = pendingRightCausalResult;
            pendingRightCausalResult = default;
            return rightResult;
        }

        private ref ArenaCausalStepResult GetMutableCausalResult(ArenaSide side)
        {
            if (side == ArenaSide.Left)
            {
                return ref pendingLeftCausalResult;
            }

            return ref pendingRightCausalResult;
        }

        private ref ArenaRoundActorStats GetMutableStats(ArenaSide side)
        {
            if (side == ArenaSide.Left)
            {
                return ref leftRoundStats;
            }

            return ref rightRoundStats;
        }

        private string BuildCausalTags(ArenaObservationSnapshot observation, ArenaActionSnapshot action)
        {
            var builder = new StringBuilder();

            switch (observation.targetType)
            {
                case ArenaTargetType.Base:
                    AppendTag(builder, "return_to_base");
                    break;
                case ArenaTargetType.Opponent:
                    AppendTag(builder, "chase_opponent");
                    break;
                default:
                    AppendTag(builder, "target_item");
                    break;
            }

            if (observation.selfHasItem)
            {
                AppendTag(builder, "self_has_item");
                AppendTag(builder, "return_to_base");
            }

            if (observation.opponentHasItem)
            {
                AppendTag(builder, "opponent_has_item");
                AppendTag(builder, "chase_opponent");
            }

            if (Vector2.Distance(observation.selfPosition, observation.opponentPosition) <= 1.1f)
            {
                AppendTag(builder, "shove_opportunity");
            }

            if (observation.wallAhead)
            {
                AppendTag(builder, "avoid_wall");
            }

            if (observation.doorOpen)
            {
                AppendTag(builder, "door_open");
            }
            else
            {
                AppendTag(builder, "door_closed");
                if (observation.shortcutBlocked)
                {
                    AppendTag(builder, "shortcut_blocked");
                }

                if (observation.detourNeeded)
                {
                    AppendTag(builder, "detour_needed");
                }
            }

            if (observation.switchActive)
            {
                AppendTag(builder, "switch_active");
            }
            else if (observation.switchDelta.sqrMagnitude <= 9f)
            {
                AppendTag(builder, "switch_available");
            }

            if (observation.movingObstacleAhead)
            {
                AppendTag(builder, "moving_obstacle_ahead");
            }

            if (roundDurationSeconds > 0f
                && observation.roundElapsedTime / roundDurationSeconds >= 0.8f)
            {
                AppendTag(builder, "time_pressure");
            }

            if (action.shovePressed)
            {
                AppendTag(builder, "shove_input");
            }

            return builder.ToString();
        }

        private static void AppendTag(StringBuilder builder, string tag)
        {
            if (builder.Length > 0)
            {
                builder.Append(',');
            }

            builder.Append(tag);
        }

        private static ArenaDqnAction ToDqnAction(ArenaActionSnapshot action)
        {
            var movingLeft = action.moveAction == ArenaMoveAction.MoveLeft;
            var movingRight = action.moveAction == ArenaMoveAction.MoveRight;
            var movingUp = action.moveAction == ArenaMoveAction.MoveUp;
            var movingDown = action.moveAction == ArenaMoveAction.MoveDown;

            if (action.shovePressed)
            {
                return ArenaDqnAction.Shove;
            }

            if (movingUp)
            {
                return ArenaDqnAction.MoveUp;
            }

            if (movingDown)
            {
                return ArenaDqnAction.MoveDown;
            }

            if (movingLeft)
            {
                return ArenaDqnAction.MoveLeft;
            }

            if (movingRight)
            {
                return ArenaDqnAction.MoveRight;
            }

            return ArenaDqnAction.Idle;
        }

        private void PrepareRoundLogFile()
        {
            if (!writeLogsToFile)
            {
                return;
            }

            Directory.CreateDirectory(outputDirectoryPath);
            currentRoundOutputPath = Path.Combine(outputDirectoryPath, $"round_{roundIndex:0000}.jsonl");
            if (File.Exists(currentRoundOutputPath))
            {
                File.Delete(currentRoundOutputPath);
            }
        }

        private void EnsureRoundLogFileReady()
        {
            if (!writeLogsToFile)
            {
                return;
            }

            if (string.IsNullOrEmpty(outputDirectoryPath))
            {
                outputDirectoryPath = Path.Combine(Application.persistentDataPath, outputDirectoryName);
            }

            if (string.IsNullOrEmpty(currentRoundOutputPath))
            {
                if (roundIndex <= 0)
                {
                    roundIndex = 1;
                    roundStartTime = Time.time;
                }

                PrepareRoundLogFile();
            }
        }

        private IEnumerator ResetRoundAfterDelay()
        {
            yield return new WaitForSeconds(roundResetDelaySeconds);
            BeginRound();
        }

        private void OnGUI()
        {
            if (!HasRequiredReferences())
            {
                return;
            }

            GUI.color = Color.white;
            GUILayout.BeginArea(new Rect(10f, 10f, 620f, 300f), GUI.skin.box);
            GUILayout.Label("Deep AI Arena Prototype");
            GUILayout.Label($"Score  Left {leftScore} : {rightScore} Right");
            GUILayout.Label($"Control  Left: {leftActorMode} / Right: {rightActorMode}");
            GUILayout.Label($"Item lane: {GetLaneName(item.transform.position.y)}");
            GUILayout.Label($"V2 env  Doors opened: {roundDoorOpenCount} / Switches: {roundSwitchActivationCount}");
            GUILayout.Label($"Route  Left: {(player != null ? player.DebugRouteName : "N/A")} / Right: {(ghost != null ? ghost.DebugRouteName : "N/A")}");
            var leftObservation = BuildObservation(ArenaSide.Left);
            var rightObservation = BuildObservation(ArenaSide.Right);
            GUILayout.Label($"Left target: {leftObservation.targetType} {FormatVector2(leftObservation.targetPosition)} "
                + $"dist {Vector2.Distance(leftObservation.selfPosition, leftObservation.targetPosition):0.00}");
            GUILayout.Label($"Right target: {rightObservation.targetType} {FormatVector2(rightObservation.targetPosition)} "
                + $"dist {Vector2.Distance(rightObservation.selfPosition, rightObservation.targetPosition):0.00}");
            GUILayout.Label($"Left env: door {(leftObservation.doorOpen ? "open" : "closed")} "
                + $"switch {(leftObservation.switchActive ? "active" : "idle")} "
                + $"movingAhead {leftObservation.movingObstacleAhead} "
                + $"detour {leftObservation.detourNeeded}");
            var ghostController = ghost != null ? ghost.GetComponent<ArenaGhostController>() : null;
            var playerGhostController = player != null ? player.GetComponent<ArenaGhostController>() : null;
            GUILayout.Label($"Runtime  Left: {FormatActorRuntime(playerGhostController, leftActorMode)} "
                + $"/ Right: {FormatActorRuntime(ghostController, rightActorMode)}");
            GUILayout.Label("Controls: WASD or arrows to move, Ctrl/E to shove.");
            GUILayout.Label("Goal: grab one of three items and reach the shared base.");
            GUILayout.EndArea();
        }

        private static string GetLaneName(float y)
        {
            if (y > 1.4f)
            {
                return "Top";
            }

            if (y < -1.2f)
            {
                return "Bottom";
            }

            return "Center";
        }

        private static string FormatVector2(Vector2 value)
        {
            return $"({value.x:0.00}, {value.y:0.00})";
        }

        private static string FormatActorRuntime(ArenaGhostController controller, ArenaActorControlMode mode)
        {
            if (controller == null || !controller.ControlActive)
            {
                return mode.ToString();
            }

            return controller.RuntimeMode.ToString();
        }

        private static int GetLaneIndex(float y)
        {
            if (y > 1.4f)
            {
                return 1;
            }

            if (y < -1.2f)
            {
                return -1;
            }

            return 0;
        }

        [System.Serializable]
        private struct ArenaStepLog
        {
            public string logType;
            public string actorSide;
            public int roundIndex;
            public float timestamp;
            public ArenaObservationSnapshot observation;
            public ArenaActionSnapshot action;
        }
    }
}
