using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(ArenaCharacterController))]
    [RequireComponent(typeof(BehaviorParameters))]
    [RequireComponent(typeof(DecisionRequester))]
    public class ArenaMlAgent : Agent
    {
        private const int ObservationSize = 36;
        private const int MoveActionCount = 5;
        private const int ShoveActionCount = 2;

        [SerializeField] private bool controlActive;
        [SerializeField] private float targetProgressRewardScale = 0.03f;
        [SerializeField] private float wallActionPenalty = -0.02f;
        [SerializeField] private float stepPenalty = -0.001f;
        [SerializeField] private int decisionPeriod = 5;

        private ArenaCharacterController controller;
        private ArenaGameManager manager;
        private bool hasPreviousObservation;
        private float previousDistanceToTarget;
        private int previousRoundIndex = -1;
        private bool episodeEndedForRound;

        public bool ControlActive => controlActive;

        public void SetControlActive(bool active)
        {
            controlActive = active;
            enabled = active;
            var requester = GetComponent<DecisionRequester>();
            if (requester != null)
            {
                requester.enabled = active;
            }
        }

        protected override void Awake()
        {
            controller = GetComponent<ArenaCharacterController>();
            ConfigureMlAgentsComponents();
            base.Awake();
        }

        private void Start()
        {
            manager ??= FindAnyObjectByType<ArenaGameManager>();
        }

        public override void Initialize()
        {
            controller ??= GetComponent<ArenaCharacterController>();
            manager ??= FindAnyObjectByType<ArenaGameManager>();
            ConfigureMlAgentsComponents();
        }

        public override void OnEpisodeBegin()
        {
            hasPreviousObservation = false;
            previousRoundIndex = manager != null ? manager.RoundIndex : -1;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            if (!TryGetObservation(out var observation))
            {
                for (var i = 0; i < ObservationSize; i++)
                {
                    sensor.AddObservation(0f);
                }

                return;
            }

            AddObservation(sensor, observation.selfPosition);
            AddObservation(sensor, observation.opponentPosition);
            AddObservation(sensor, observation.itemPosition);
            AddObservation(sensor, observation.basePosition);
            sensor.AddObservation(observation.selfHasItem ? 1f : 0f);
            sensor.AddObservation(observation.opponentHasItem ? 1f : 0f);
            AddObservation(sensor, observation.itemDelta);
            AddObservation(sensor, observation.baseDelta);
            AddObservation(sensor, observation.opponentDelta);
            sensor.AddObservation(observation.distanceToItem);
            sensor.AddObservation(observation.distanceToBase);
            sensor.AddObservation(observation.distanceToOpponent);
            sensor.AddObservation(observation.distanceToTarget);
            AddObservation(sensor, observation.targetPosition);
            sensor.AddObservation((float)observation.targetType);
            sensor.AddObservation(observation.wallAhead ? 1f : 0f);
            sensor.AddObservation(observation.roundElapsedTime);
            AddObservation(sensor, observation.doorDelta);
            sensor.AddObservation(observation.doorOpen ? 1f : 0f);
            AddObservation(sensor, observation.switchDelta);
            sensor.AddObservation(observation.switchActive ? 1f : 0f);
            AddObservation(sensor, observation.movingObstacleDelta);
            AddObservation(sensor, observation.movingObstacleVelocity);
            sensor.AddObservation(observation.movingObstacleAhead ? 1f : 0f);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (!controlActive || !TryGetObservation(out var observation))
            {
                return;
            }

            if (previousRoundIndex != manager.RoundIndex)
            {
                hasPreviousObservation = false;
                previousRoundIndex = manager.RoundIndex;
                episodeEndedForRound = false;
            }

            var moveAction = Mathf.Clamp(actions.DiscreteActions[0], 0, MoveActionCount - 1);
            var shoveAction = Mathf.Clamp(actions.DiscreteActions[1], 0, ShoveActionCount - 1);

            AddReward(manager.ConsumeMlAgentReward(controller.Side));

            if (hasPreviousObservation)
            {
                var progress = previousDistanceToTarget - observation.distanceToTarget;
                AddReward(Mathf.Clamp(progress * targetProgressRewardScale, -0.05f, 0.05f));
            }

            AddReward(stepPenalty);
            if (observation.wallAhead && moveAction != 0)
            {
                AddReward(wallActionPenalty);
            }

            if (manager.IsRoundTransitioning)
            {
                controller.SetGhostInput(Vector2.zero, false, "MLAgentsDone");
                EndCurrentRoundEpisode();
                return;
            }

            controller.SetGhostInput(ToMoveVector(moveAction), shoveAction == 1, "MLAgents");
            previousDistanceToTarget = observation.distanceToTarget;
            hasPreviousObservation = true;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            if (controller == null || !controlActive)
            {
                discreteActions[0] = 0;
                discreteActions[1] = 0;
                return;
            }

            var snapshot = controller.GetCurrentActionSnapshot();
            discreteActions[0] = ToDiscreteMoveAction(snapshot.moveAction);
            discreteActions[1] = snapshot.shovePressed ? 1 : 0;
        }

        private void Update()
        {
            if (!controlActive || manager == null)
            {
                return;
            }

            if (previousRoundIndex != manager.RoundIndex)
            {
                previousRoundIndex = manager.RoundIndex;
                episodeEndedForRound = false;
            }

            if (!manager.IsRoundTransitioning)
            {
                return;
            }

            AddReward(manager.ConsumeMlAgentReward(controller.Side));
            EndCurrentRoundEpisode();
        }

        private void EndCurrentRoundEpisode()
        {
            if (manager == null || episodeEndedForRound)
            {
                return;
            }

            episodeEndedForRound = true;
            EndEpisode();
        }

        private bool TryGetObservation(out ArenaObservationSnapshot observation)
        {
            controller ??= GetComponent<ArenaCharacterController>();
            manager ??= FindAnyObjectByType<ArenaGameManager>();
            if (manager == null || controller == null)
            {
                observation = default;
                return false;
            }

            observation = manager.BuildObservation(controller.Side);
            return true;
        }

        private void ConfigureMlAgentsComponents()
        {
            var behavior = GetComponent<BehaviorParameters>();
            behavior.BehaviorName = "ArenaMlAgent";
            behavior.TeamId = controller != null && controller.Side == ArenaSide.Right ? 1 : 0;
            behavior.BehaviorType = BehaviorType.Default;
            behavior.BrainParameters.VectorObservationSize = ObservationSize;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(MoveActionCount, ShoveActionCount);

            var requester = GetComponent<DecisionRequester>();
            requester.DecisionPeriod = Mathf.Max(1, decisionPeriod);
            requester.DecisionStep = 0;
            requester.TakeActionsBetweenDecisions = true;
        }

        private static void AddObservation(VectorSensor sensor, Vector2 value)
        {
            sensor.AddObservation(value.x);
            sensor.AddObservation(value.y);
        }

        private static Vector2 ToMoveVector(int action)
        {
            return action switch
            {
                1 => Vector2.up,
                2 => Vector2.down,
                3 => Vector2.left,
                4 => Vector2.right,
                _ => Vector2.zero
            };
        }

        private static int ToDiscreteMoveAction(ArenaMoveAction action)
        {
            return action switch
            {
                ArenaMoveAction.MoveUp => 1,
                ArenaMoveAction.MoveDown => 2,
                ArenaMoveAction.MoveLeft => 3,
                ArenaMoveAction.MoveRight => 4,
                _ => 0
            };
        }
    }
}
