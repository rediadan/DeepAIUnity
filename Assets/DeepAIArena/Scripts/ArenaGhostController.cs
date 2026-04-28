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

        public ArenaGhostPolicyMode PolicyMode => policyMode;
        public ArenaGhostRuntimeMode RuntimeMode { get; private set; } = ArenaGhostRuntimeMode.RuleBased;

        private void Awake()
        {
            controller = GetComponent<ArenaCharacterController>();
            onnxPolicy ??= GetComponent<ArenaGhostOnnxPolicy>();
        }

        private void Update()
        {
            manager ??= FindAnyObjectByType<ArenaGameManager>();
            if (manager == null)
            {
                return;
            }

            if (policyMode == ArenaGhostPolicyMode.OnnxInference)
            {
                if (onnxPolicy != null
                    && onnxPolicy.TryEvaluate(manager.BuildObservation(ArenaSide.Right), out var modelAction))
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
                    && onnxPolicy.TryEvaluateDqn(manager.BuildObservation(ArenaSide.Right), out var modelAction))
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

            var target = GetTargetPoint(out var routeName);
            var delta = target - (Vector2)transform.position;
            var move = delta.sqrMagnitude > 0.08f ? delta.normalized : Vector2.zero;
            var shove = ShouldShove();
            controller.SetGhostInput(move, shove, routeName);
        }

        private Vector2 GetTargetPoint(out string routeName)
        {
            if (controller.HasItem)
            {
                routeName = "Escape";
                return manager.GetSharedBasePoint();
            }

            var itemPosition = manager.Item.transform.position;
            var playerPosition = manager.Player.transform.position;
            var preferIntercept = manager.Player.HasItem;

            if (itemPosition.y > 1.4f)
            {
                routeName = "Top";
                return preferIntercept ? playerPosition : itemPosition;
            }

            if (itemPosition.y < -1.2f)
            {
                routeName = "Bottom";
                return preferIntercept ? playerPosition : itemPosition;
            }

            routeName = "Center";
            return preferIntercept ? playerPosition : itemPosition;
        }

        private bool ShouldShove()
        {
            var player = manager.Player;
            if (player == null)
            {
                return false;
            }

            return Vector2.Distance(transform.position, player.transform.position) < 1.1f
                && player.HasItem;
        }
    }
}
