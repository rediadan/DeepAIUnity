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

    [System.Serializable]
    public struct ArenaObservationSnapshot
    {
        public Vector2 selfPosition;
        public Vector2 opponentPosition;
        public Vector2 itemPosition;
        public Vector2 selfVelocity;
        public Vector2 opponentVelocity;
        public Vector2 selfBasePosition;
        public Vector2 opponentBasePosition;
        public bool selfHasItem;
        public bool opponentHasItem;
        public bool isGrounded;
        public bool opponentGrounded;
        public int itemLane;
    }

    [System.Serializable]
    public struct ArenaActionSnapshot
    {
        public ArenaMoveAction moveAction;
        public bool jumpPressed;
        public bool dropPressed;
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

    public class ArenaGameManager : MonoBehaviour
    {
        private readonly Vector3[] itemSpawnPoints =
        {
            new Vector3(0f, 2.95f, 0f),
            new Vector3(0f, 1.95f, 0f),
            new Vector3(0f, 0.95f, 0f)
        };

        [SerializeField] private ArenaCharacterController player;
        [SerializeField] private ArenaCharacterController ghost;
        [SerializeField] private ArenaItem item;
        [SerializeField] private ArenaBaseZone leftBase;
        [SerializeField] private ArenaBaseZone rightBase;
        [SerializeField] private bool writeLogsToFile = true;
        [SerializeField] private string outputFileName = "arena_training_log.jsonl";

        private int leftScore;
        private int rightScore;
        private int roundIndex;
        private bool roundTransition;
        private string outputPath;

        public ArenaCharacterController Player => player;
        public ArenaCharacterController Ghost => ghost;
        public ArenaItem Item => item;
        public Vector3[] ItemSpawnPoints => itemSpawnPoints;
        public ArenaSide LastScoringSide { get; private set; }

        private void Awake()
        {
            EnsureReferences();
            outputPath = Path.Combine(Application.persistentDataPath, outputFileName);
            if (writeLogsToFile && File.Exists(outputPath))
            {
                File.Delete(outputPath);
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
            ArenaBaseZone playerBase,
            ArenaBaseZone ghostBase)
        {
            player = playerController;
            ghost = ghostController;
            item = arenaItem;
            leftBase = playerBase;
            rightBase = ghostBase;

            player.Initialize(this);
            ghost.Initialize(this);
            playerBase.Initialize(this);
            ghostBase.Initialize(this);
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

            if (leftBase == null || rightBase == null)
            {
                var bases = GetComponentsInChildren<ArenaBaseZone>(true);
                foreach (var arenaBase in bases)
                {
                    if (arenaBase == null)
                    {
                        continue;
                    }

                    if (arenaBase.Side == ArenaSide.Left)
                    {
                        leftBase ??= arenaBase;
                    }
                    else if (arenaBase.Side == ArenaSide.Right)
                    {
                        rightBase ??= arenaBase;
                    }
                }
            }
        }

        private bool HasRequiredReferences()
        {
            return player != null && ghost != null && item != null && leftBase != null && rightBase != null;
        }

        private void InitializeRuntimeReferences()
        {
            player.Initialize(this);
            ghost.Initialize(this);
            leftBase.Initialize(this);
            rightBase.Initialize(this);
            item.Initialize(this);
        }

        public Vector3 GetSpawnPoint(ArenaSide side)
        {
            return side == ArenaSide.Left ? new Vector3(-12f, 1.25f, 0f) : new Vector3(12f, 1.25f, 0f);
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

            LogStep(ArenaSide.Left, roundIndex, BuildObservation(ArenaSide.Left), player.GetCurrentActionSnapshot());
        }

        public ArenaObservationSnapshot BuildObservation(ArenaSide side)
        {
            var self = side == ArenaSide.Left ? player : ghost;
            var opponent = side == ArenaSide.Left ? ghost : player;
            var selfBase = side == ArenaSide.Left ? leftBase : rightBase;
            var opponentBase = side == ArenaSide.Left ? rightBase : leftBase;

            return new ArenaObservationSnapshot
            {
                selfPosition = self.transform.position,
                opponentPosition = opponent.transform.position,
                itemPosition = item.transform.position,
                selfVelocity = self.Body != null ? self.Body.linearVelocity : Vector2.zero,
                opponentVelocity = opponent.Body != null ? opponent.Body.linearVelocity : Vector2.zero,
                selfBasePosition = selfBase.transform.position,
                opponentBasePosition = opponentBase.transform.position,
                selfHasItem = self.HasItem,
                opponentHasItem = opponent.HasItem,
                isGrounded = self.IsGrounded(),
                opponentGrounded = opponent.IsGrounded(),
                itemLane = GetLaneIndex(item.transform.position.y)
            };
        }

        public void ReportReward(ArenaRewardEventType eventType, ArenaSide actorSide, float rewardDelta, string note)
        {
            if (!writeLogsToFile)
            {
                return;
            }

            var rewardEvent = new ArenaRewardEvent
            {
                eventType = eventType,
                actorSide = actorSide,
                rewardDelta = rewardDelta,
                timestamp = Time.time,
                roundIndex = roundIndex,
                note = note
            };

            var json = JsonUtility.ToJson(rewardEvent);
            File.AppendAllText(outputPath, json + "\n", Encoding.UTF8);
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

            var json = JsonUtility.ToJson(new ArenaStepLog
            {
                actorSide = actorSide.ToString(),
                roundIndex = currentRoundIndex,
                observation = observation,
                action = action,
                timestamp = Time.time
            });

            File.AppendAllText(outputPath, json + "\n", Encoding.UTF8);
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
            GUILayout.Label("Controls: A/D or Left/Right to move, Space to jump.");
            GUILayout.Label("Goal: grab the central item and bring it back to your base.");
            GUILayout.EndArea();
        }

        private static string GetLaneName(float y)
        {
            if (y > 2.4f)
            {
                return "Top";
            }

            if (y < 1.4f)
            {
                return "Bottom";
            }

            return "Center";
        }

        private static int GetLaneIndex(float y)
        {
            if (y > 2.4f)
            {
                return 1;
            }

            if (y < 1.4f)
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
