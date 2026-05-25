using UnityEngine;

namespace DeepAIArena
{
    public enum ArenaGhostPolicyMode
    {
        RuleBased,
        OnnxInference,
        DqnInference
    }

    public enum ArenaGhostRuntimeMode
    {
        RuleBased,
        OnnxInference,
        DqnInference,
        OnnxFailed,
        DqnFailed,
        OnnxFallback
    }

    [RequireComponent(typeof(ArenaCharacterController))]
    public class ArenaGhostController : MonoBehaviour
    {
        [Header("Ghost Policy")]
        [Tooltip("Rule-Based keeps handcrafted logic. ONNX Inference runs BC. DQN Inference runs q-value policy.")]
        [SerializeField] private ArenaGhostPolicyMode policyMode = ArenaGhostPolicyMode.RuleBased;
        [SerializeField] private ArenaGhostOnnxPolicy onnxPolicy;
        [SerializeField] private bool allowRuleFallbackOnInferenceFailure;

        private ArenaCharacterController controller;
        private ArenaGameManager manager;
        private bool controlActive = true;

        public ArenaGhostPolicyMode PolicyMode => policyMode;
        public ArenaGhostRuntimeMode RuntimeMode { get; private set; } = ArenaGhostRuntimeMode.RuleBased;
        public bool ControlActive => controlActive;

        public void SetPolicyMode(ArenaGhostPolicyMode mode)
        {
            policyMode = mode;
        }

        public void SetControlActive(bool active)
        {
            controlActive = active;
            if (!active)
            {
                RuntimeMode = ArenaGhostRuntimeMode.RuleBased;
            }
        }

        private void Awake()
        {
            controller = GetComponent<ArenaCharacterController>();
            onnxPolicy ??= GetComponent<ArenaGhostOnnxPolicy>();
        }

        private void Update()
        {
            if (!controlActive)
            {
                return;
            }

            manager ??= FindAnyObjectByType<ArenaGameManager>();
            if (manager == null)
            {
                return;
            }

            if (policyMode == ArenaGhostPolicyMode.OnnxInference)
            {
                if (onnxPolicy != null
                    && onnxPolicy.TryEvaluate(manager.BuildObservation(controller.Side), out var modelAction))
                {
                    RuntimeMode = ArenaGhostRuntimeMode.OnnxInference;
                    controller.SetGhostInput(
                        modelAction.move,
                        modelAction.shove,
                        modelAction.routeName);
                    return;
                }

                if (!allowRuleFallbackOnInferenceFailure)
                {
                    RuntimeMode = ArenaGhostRuntimeMode.OnnxFailed;
                    controller.SetGhostInput(Vector2.zero, false, "OnnxFailed");
                    return;
                }

                RuntimeMode = ArenaGhostRuntimeMode.OnnxFallback;
            }
            else if (policyMode == ArenaGhostPolicyMode.DqnInference)
            {
                if (onnxPolicy != null
                    && onnxPolicy.TryEvaluateDqn(manager.BuildObservation(controller.Side), out var modelAction))
                {
                    RuntimeMode = ArenaGhostRuntimeMode.DqnInference;
                    controller.SetGhostInput(
                        modelAction.move,
                        modelAction.shove,
                        modelAction.routeName);
                    return;
                }

                if (!allowRuleFallbackOnInferenceFailure)
                {
                    RuntimeMode = ArenaGhostRuntimeMode.DqnFailed;
                    controller.SetGhostInput(Vector2.zero, false, "DqnFailed");
                    return;
                }

                RuntimeMode = ArenaGhostRuntimeMode.OnnxFallback;
            }
            else
            {
                RuntimeMode = ArenaGhostRuntimeMode.RuleBased;
            }

            var observation = manager.BuildObservation(controller.Side);
            var target = GetTargetPoint(observation, out var routeName);
            var delta = target - (Vector2)transform.position;
            var move = delta.sqrMagnitude > 0.08f ? delta.normalized : Vector2.zero;
            if (observation.movingObstacleAhead && move.sqrMagnitude > 0.0001f)
            {
                move = ChooseObstacleAvoidanceMove(move);
                routeName = "AvoidMovingObstacle";
            }

            var shove = ShouldShove();
            controller.SetGhostInput(move, shove, routeName);
        }

        private Vector2 GetTargetPoint(ArenaObservationSnapshot observation, out string routeName)
        {
            if (!observation.doorOpen
                && !observation.switchActive
                && observation.doorDelta.sqrMagnitude <= 9f
                && observation.switchDelta.sqrMagnitude <= 25f)
            {
                routeName = "Switch";
                return observation.selfPosition + observation.switchDelta;
            }

            if (controller.HasItem)
            {
                routeName = "Escape";
                return manager.GetSharedBasePoint();
            }

            var itemPosition = manager.Item.transform.position;
            var opponent = controller.Side == ArenaSide.Left ? manager.Ghost : manager.Player;
            var opponentPosition = opponent.transform.position;
            var preferIntercept = opponent.HasItem;

            if (itemPosition.y > 1.4f)
            {
                routeName = "Top";
                return preferIntercept ? opponentPosition : itemPosition;
            }

            if (itemPosition.y < -1.2f)
            {
                routeName = "Bottom";
                return preferIntercept ? opponentPosition : itemPosition;
            }

            routeName = "Center";
            return preferIntercept ? opponentPosition : itemPosition;
        }

        private static Vector2 ChooseObstacleAvoidanceMove(Vector2 move)
        {
            var perpendicular = new Vector2(-move.y, move.x);
            return perpendicular.sqrMagnitude > 0.0001f ? perpendicular.normalized : Vector2.up;
        }

        private bool ShouldShove()
        {
            var opponent = controller.Side == ArenaSide.Left ? manager.Ghost : manager.Player;
            if (opponent == null)
            {
                return false;
            }

            return Vector2.Distance(transform.position, opponent.transform.position) < 1.1f
                && opponent.HasItem;
        }
    }
}
