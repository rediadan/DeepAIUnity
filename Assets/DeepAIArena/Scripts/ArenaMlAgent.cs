using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.InputSystem;

namespace DeepAIArena
{
    [RequireComponent(typeof(ArenaCharacterController))]
    [RequireComponent(typeof(BehaviorParameters))]
    [RequireComponent(typeof(DecisionRequester))]
    public class ArenaMlAgent : Agent
    {
        private const int ObservationSize = 38;
        private const int MoveActionCount = 5;
        private const int ShoveActionCount = 2;

        [SerializeField] private bool controlActive;
        [SerializeField] private int decisionPeriod = 5;
        [SerializeField] private bool useManualInputInHeuristic = true;

        private ArenaCharacterController controller;
        private ArenaGameManager manager;
        private bool hasPreviousObservation;
        private ArenaObservationSnapshot previousObservation;
        private ArenaDqnAction previousAction;
        private int previousRoundIndex = -1;
        private bool episodeEndedForRound;
        private int fixedDecisionTick;

        public bool ControlActive => controlActive;

        public void SetControlActive(bool active)
        {
            controller ??= GetComponent<ArenaCharacterController>();
            manager ??= FindAnyObjectByType<ArenaGameManager>();
            DisableSiblingBaseAgents();
            ConfigureMlAgentsComponents();

            controlActive = active;
            enabled = active;
            var requester = GetComponent<DecisionRequester>();
            if (requester != null)
            {
                requester.enabled = active;
            }

            if (active)
            {
                fixedDecisionTick = 0;
                RequestDecision();
            }
        }

        protected override void Awake()
        {
            controller = GetComponent<ArenaCharacterController>();
            DisableSiblingBaseAgents();
            ConfigureMlAgentsComponents();
            base.Awake();
        }

        private void Start()
        {
            DisableSiblingBaseAgents();
            manager ??= FindAnyObjectByType<ArenaGameManager>();
            ConfigureMlAgentsComponents();
        }

        public override void Initialize()
        {
            controller ??= GetComponent<ArenaCharacterController>();
            manager ??= FindAnyObjectByType<ArenaGameManager>();
            DisableSiblingBaseAgents();
            ConfigureMlAgentsComponents();
        }

        public override void OnEpisodeBegin()
        {
            hasPreviousObservation = false;
            previousRoundIndex = manager != null ? manager.RoundIndex : -1;
            previousAction = ArenaDqnAction.Idle;
            episodeEndedForRound = false;
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
            sensor.AddObservation(observation.shortcutBlocked ? 1f : 0f);
            sensor.AddObservation(observation.detourNeeded ? 1f : 0f);
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
            var currentAction = ToDqnAction(moveAction, shoveAction);

            AddReward(manager.ConsumeMlAgentReward(controller.Side));

            if (hasPreviousObservation)
            {
                AddReward(manager.ComputeSharedAgentShapingReward(previousObservation, observation, previousAction));
            }

            if (manager.IsRoundTransitioning)
            {
                controller.SetGhostInput(Vector2.zero, false, "MLAgentsDone");
                EndCurrentRoundEpisode();
                return;
            }

            controller.SetGhostInput(ToMoveVector(moveAction), shoveAction == 1, "MLAgents");
            previousObservation = observation;
            previousAction = currentAction;
            hasPreviousObservation = true;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            controller ??= GetComponent<ArenaCharacterController>();
            if (controller == null || !controlActive)
            {
                discreteActions[0] = 0;
                discreteActions[1] = 0;
                return;
            }

            if (useManualInputInHeuristic)
            {
                discreteActions[0] = ReadManualMoveAction();
                discreteActions[1] = IsManualShovePressed() ? 1 : 0;
                return;
            }

            if (TryGetObservation(out var observation))
            {
                var target = GetHeuristicTarget(observation);
                var delta = target - observation.selfPosition;
                discreteActions[0] = ToDiscreteMoveAction(delta);
                discreteActions[1] = observation.opponentDelta.sqrMagnitude <= 1.21f && observation.opponentHasItem ? 1 : 0;
                return;
            }

            discreteActions[0] = 0;
            discreteActions[1] = 0;
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

        private void FixedUpdate()
        {
            if (!controlActive || !isActiveAndEnabled)
            {
                return;
            }

            fixedDecisionTick++;
            if (fixedDecisionTick < Mathf.Max(1, decisionPeriod))
            {
                return;
            }

            fixedDecisionTick = 0;
            RequestDecision();
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
            DisableSiblingBaseAgents();

            var behavior = GetComponent<BehaviorParameters>();
            behavior.BehaviorName = "ArenaMlAgent";
            behavior.TeamId = controller != null && controller.Side == ArenaSide.Right ? 1 : 0;
            behavior.BrainParameters.VectorObservationSize = ObservationSize;
            behavior.BrainParameters.NumStackedVectorObservations = 1;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(MoveActionCount, ShoveActionCount);

            var requester = GetComponent<DecisionRequester>();
            requester.DecisionPeriod = Mathf.Max(1, decisionPeriod);
            requester.DecisionStep = 0;
            requester.TakeActionsBetweenDecisions = true;
        }

        private void DisableSiblingBaseAgents()
        {
            var agents = GetComponents<Agent>();
            foreach (var agent in agents)
            {
                if (agent != null && agent != this && agent.GetType() == typeof(Agent))
                {
                    agent.enabled = false;
                }
            }
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

        private static ArenaDqnAction ToDqnAction(int moveAction, int shoveAction)
        {
            if (shoveAction == 1)
            {
                return ArenaDqnAction.Shove;
            }

            return moveAction switch
            {
                1 => ArenaDqnAction.MoveUp,
                2 => ArenaDqnAction.MoveDown,
                3 => ArenaDqnAction.MoveLeft,
                4 => ArenaDqnAction.MoveRight,
                _ => ArenaDqnAction.Idle
            };
        }

        private static int ReadManualMoveAction()
        {
            var move = Vector2.zero;
            var keyboard = Keyboard.current;
            if (keyboard != null)
            {
                var horizontal = 0f;
                var vertical = 0f;
                if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed)
                {
                    horizontal -= 1f;
                }

                if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed)
                {
                    horizontal += 1f;
                }

                if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed)
                {
                    vertical -= 1f;
                }

                if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed)
                {
                    vertical += 1f;
                }

                move = new Vector2(horizontal, vertical);
            }

            var gamepad = Gamepad.current;
            if (gamepad != null)
            {
                var stick = gamepad.leftStick.ReadValue();
                if (stick.sqrMagnitude > 0.01f)
                {
                    move = stick;
                }
            }

            return ToDiscreteMoveAction(move);
        }

        private static bool IsManualShovePressed()
        {
            var keyboard = Keyboard.current;
            if (keyboard != null
                && (keyboard.leftCtrlKey.isPressed
                    || keyboard.rightCtrlKey.isPressed
                    || keyboard.eKey.isPressed
                    || keyboard.spaceKey.isPressed))
            {
                return true;
            }

            var gamepad = Gamepad.current;
            return gamepad != null && gamepad.rightShoulder.isPressed;
        }

        private static Vector2 GetHeuristicTarget(ArenaObservationSnapshot observation)
        {
            if (!observation.doorOpen
                && !observation.switchActive
                && observation.shortcutBlocked
                && observation.switchDelta.sqrMagnitude <= 36f)
            {
                return observation.selfPosition + observation.switchDelta;
            }

            if (observation.selfHasItem)
            {
                return observation.basePosition;
            }

            return observation.opponentHasItem ? observation.opponentPosition : observation.itemPosition;
        }

        private static int ToDiscreteMoveAction(Vector2 direction)
        {
            if (direction.sqrMagnitude < 0.08f)
            {
                return 0;
            }

            if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y))
            {
                return direction.x < 0f ? 3 : 4;
            }

            return direction.y < 0f ? 2 : 1;
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
