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
        RoundTimeout
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
        MlAgents
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
        [SerializeField] private float dqnStepPenalty = -0.01f;
        [SerializeField] private float dqnTargetProgressRewardScale = 0.05f;
        [SerializeField] private float dqnWallActionPenalty = -0.02f;

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
        private ArenaCausalStepResult pendingLeftCausalResult;
        private ArenaCausalStepResult pendingRightCausalResult;
        private ArenaRoundActorStats leftRoundStats;
        private ArenaRoundActorStats rightRoundStats;
        private int roundDoorOpenCount;
        private int roundSwitchActivationCount;
        private bool hasPreviousPlayerDqnStep;
        private ArenaObservationSnapshot previousPlayerObservation;
        private ArenaActionSnapshot previousPlayerAction;

        public ArenaCharacterController Player => player;
        public ArenaCharacterController Ghost => ghost;
        public ArenaItem Item => item;
        public ArenaDoor[] ArenaDoors => arenaDoors;
        public ArenaSwitch[] ArenaSwitches => arenaSwitches;
        public ArenaMovingObstacle[] MovingObstacles => movingObstacles;
        public Vector3[] ItemSpawnPoints => itemSpawnPoints;
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
        }

        private void ConfigureActorControl(ArenaCharacterController actor, ArenaActorControlMode mode)
        {
            if (actor == null)
            {
                return;
            }

            var useAi = mode != ArenaActorControlMode.Human || actor.IsGhost;
            actor.IsGhost = useAi;

            var ghostController = actor.GetComponent<ArenaGhostController>();
            var onnxPolicy = actor.GetComponent<ArenaGhostOnnxPolicy>();
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
                    onnxPolicy.enabled = true;
                }

                if (mlAgent != null)
                {
                    mlAgent.SetControlActive(false);
                }

                actor.DebugRouteName = "Human";
                return;
            }

            ghostController ??= actor.gameObject.AddComponent<ArenaGhostController>();
            if (mode == ArenaActorControlMode.MlAgents)
            {
                mlAgent ??= actor.gameObject.AddComponent<ArenaMlAgent>();
                mlAgent.SetControlActive(true);
                ghostController.enabled = true;
                ghostController.SetControlActive(false);
                actor.DebugRouteName = "MLAgents";
                return;
            }

            if (mlAgent != null)
            {
                mlAgent.SetControlActive(false);
            }

            if (mode != ArenaActorControlMode.RuleBased)
            {
                onnxPolicy ??= actor.gameObject.AddComponent<ArenaGhostOnnxPolicy>();
                onnxPolicy.enabled = true;
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
            pendingLeftCausalResult = default;
            pendingRightCausalResult = default;
            hasPreviousPlayerDqnStep = false;
            ResetRoundStats();
            ResetEnvironmentObjects();
            PrepareRoundLogFile();

            player.ResetActor(GetSpawnPoint(ArenaSide.Left));
            ghost.ResetActor(GetSpawnPoint(ArenaSide.Right));
            item.ResetToSpawn(itemSpawnPoints[Random.Range(0, itemSpawnPoints.Length)]);
            ReportReward(ArenaRewardEventType.RoundStart, ArenaSide.Left, 0f, "round_start");
            ReportReward(ArenaRewardEventType.RoundStart, ArenaSide.Right, 0f, "round_start");
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
                ReportReward(ArenaRewardEventType.RoundTimeout, ArenaSide.Left, 0f, "round_timeout");
                ReportReward(ArenaRewardEventType.RoundTimeout, ArenaSide.Right, 0f, "round_timeout");
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
            var nearestSwitch = FindNearestSwitch(selfPosition);
            var nearestMovingObstacle = FindNearestMovingObstacle(selfPosition);
            var doorPosition = nearestDoor != null ? (Vector2)nearestDoor.transform.position : selfPosition;
            var switchPosition = nearestSwitch != null ? (Vector2)nearestSwitch.transform.position : selfPosition;
            var movingObstaclePosition = nearestMovingObstacle != null
                ? (Vector2)nearestMovingObstacle.transform.position
                : selfPosition;
            var movingObstacleDelta = movingObstaclePosition - selfPosition;
            var movingObstacleVelocity = nearestMovingObstacle != null ? nearestMovingObstacle.Velocity : Vector2.zero;

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
                movingObstacleAhead = IsMovingObstacleAhead(selfPosition, targetPosition, movingObstacleDelta)
            };
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

        private ArenaSwitch FindNearestSwitch(Vector2 position)
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

        public void ReportDoorOpened(ArenaDoor arenaDoor)
        {
            if (arenaDoor == null || roundTransition)
            {
                return;
            }

            roundDoorOpenCount++;
        }

        public void ReportSwitchActivated(ArenaSwitch arenaSwitch)
        {
            if (arenaSwitch == null || roundTransition)
            {
                return;
            }

            roundSwitchActivationCount++;
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
                var reward = ConsumePendingDqnReward(ArenaSide.Left)
                    + ComputeShapingReward(previousPlayerObservation, currentObservation, previousPlayerAction);
                var done = roundTransition;
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
            var previousDistance = Vector2.Distance(previousObservation.selfPosition, previousObservation.targetPosition);
            var currentDistance = Vector2.Distance(currentObservation.selfPosition, currentObservation.targetPosition);
            var progressReward = Mathf.Clamp(
                (previousDistance - currentDistance) * dqnTargetProgressRewardScale,
                -0.1f,
                0.1f);
            var wallPenalty = previousObservation.wallAhead
                && previousAction.moveAction != ArenaMoveAction.Idle
                ? dqnWallActionPenalty
                : 0f;

            return dqnStepPenalty + progressReward + wallPenalty;
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
                if (observation.doorDelta.sqrMagnitude <= 6.25f)
                {
                    AppendTag(builder, "shortcut_blocked");
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
                + $"movingAhead {leftObservation.movingObstacleAhead}");
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
