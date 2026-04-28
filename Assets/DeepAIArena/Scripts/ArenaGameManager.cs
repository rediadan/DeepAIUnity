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
        FellOffMap
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
        public Vector2 targetPosition;
        public ArenaTargetType targetType;
        public bool wallAhead;
        public float roundElapsedTime;
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
        [SerializeField] private bool writeLogsToFile = true;
        [SerializeField] private string outputDirectoryName = "arena_training_rounds";
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
        private float pendingLeftDqnReward;
        private float pendingRightDqnReward;
        private bool hasPreviousPlayerDqnStep;
        private ArenaObservationSnapshot previousPlayerObservation;
        private ArenaActionSnapshot previousPlayerAction;

        public ArenaCharacterController Player => player;
        public ArenaCharacterController Ghost => ghost;
        public ArenaItem Item => item;
        public Vector3[] ItemSpawnPoints => itemSpawnPoints;
        public ArenaSide LastScoringSide { get; private set; }

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

            roundTransition = true;
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
            StartCoroutine(ResetRoundAfterDelay());
        }

        public void BeginRound()
        {
            roundTransition = false;
            roundIndex++;
            roundStartTime = Time.time;
            pendingLeftDqnReward = 0f;
            pendingRightDqnReward = 0f;
            hasPreviousPlayerDqnStep = false;
            PrepareRoundLogFile();

            player.ResetActor(GetSpawnPoint(ArenaSide.Left));
            ghost.ResetActor(GetSpawnPoint(ArenaSide.Right));
            item.ResetToSpawn(itemSpawnPoints[Random.Range(0, itemSpawnPoints.Length)]);
            ReportReward(ArenaRewardEventType.RoundStart, ArenaSide.Left, 0f, "round_start");
            ReportReward(ArenaRewardEventType.RoundStart, ArenaSide.Right, 0f, "round_start");
        }

        private void Update()
        {
            if (!HasRequiredReferences())
            {
                return;
            }

            var observation = BuildObservation(ArenaSide.Left);
            var action = player.GetCurrentActionSnapshot();
            LogStep(ArenaSide.Left, roundIndex, observation, action);
            LogDqnTransitionIfReady(observation, action);
        }

        public ArenaObservationSnapshot BuildObservation(ArenaSide side)
        {
            var self = side == ArenaSide.Left ? player : ghost;
            var opponent = side == ArenaSide.Left ? ghost : player;
            var basePosition = sharedBase != null ? (Vector2)sharedBase.transform.position : Vector2.zero;
            var targetType = ResolveTargetType(self, opponent);
            var targetPosition = ResolveTargetPosition(targetType, opponent, basePosition);

            return new ArenaObservationSnapshot
            {
                selfPosition = self.transform.position,
                opponentPosition = opponent.transform.position,
                itemPosition = item.transform.position,
                basePosition = basePosition,
                selfHasItem = self.HasItem,
                opponentHasItem = opponent.HasItem,
                itemDelta = (Vector2)item.transform.position - (Vector2)self.transform.position,
                baseDelta = basePosition - (Vector2)self.transform.position,
                opponentDelta = (Vector2)opponent.transform.position - (Vector2)self.transform.position,
                targetPosition = targetPosition,
                targetType = targetType,
                wallAhead = self.IsWallAhead(),
                roundElapsedTime = Time.time - roundStartTime
            };
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

            AddPendingDqnReward(actorSide, rewardDelta);

            var json = JsonUtility.ToJson(rewardEvent);
            File.AppendAllText(currentRoundOutputPath, json + "\n", Encoding.UTF8);
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
                actorSide = actorSide.ToString(),
                roundIndex = currentRoundIndex,
                observation = observation,
                action = action,
                timestamp = Time.time - roundStartTime
            });

            File.AppendAllText(currentRoundOutputPath, json + "\n", Encoding.UTF8);
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
            yield return new WaitForSeconds(1.25f);
            BeginRound();
        }

        private void OnGUI()
        {
            if (!HasRequiredReferences())
            {
                return;
            }

            GUI.color = Color.white;
            GUILayout.BeginArea(new Rect(10f, 10f, 500f, 180f), GUI.skin.box);
            GUILayout.Label("Deep AI Arena Prototype");
            GUILayout.Label($"Score  Player {leftScore} : {rightScore} Ghost");
            GUILayout.Label($"Item lane: {GetLaneName(item.transform.position.y)}");
            GUILayout.Label($"Ghost route: {(ghost != null ? ghost.DebugRouteName : "N/A")}");
            var ghostController = ghost != null ? ghost.GetComponent<ArenaGhostController>() : null;
            GUILayout.Label($"Ghost mode: {(ghostController != null ? ghostController.PolicyMode.ToString() : "N/A")}");
            GUILayout.Label($"Ghost runtime: {(ghostController != null ? ghostController.RuntimeMode.ToString() : "N/A")}");
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
            public string actorSide;
            public int roundIndex;
            public float timestamp;
            public ArenaObservationSnapshot observation;
            public ArenaActionSnapshot action;
        }
    }
}
