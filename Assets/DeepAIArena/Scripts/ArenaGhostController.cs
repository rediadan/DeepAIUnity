using UnityEngine;

namespace DeepAIArena
{
    public enum ArenaGhostPolicyMode
    {
        RuleBased,
        OnnxInference
    }

    [RequireComponent(typeof(ArenaCharacterController))]
    public class ArenaGhostController : MonoBehaviour
    {
        [Header("Ghost Policy")]
        [Tooltip("Rule-Based keeps the handcrafted Ghost logic. ONNX Input runs the trained model.")]
        [SerializeField] private ArenaGhostPolicyMode policyMode = ArenaGhostPolicyMode.RuleBased;
        [SerializeField] private ArenaGhostOnnxPolicy onnxPolicy;

        private ArenaCharacterController controller;
        private ArenaGameManager manager;

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

            if (policyMode == ArenaGhostPolicyMode.OnnxInference
                && onnxPolicy != null
                && onnxPolicy.TryEvaluate(manager.BuildObservation(ArenaSide.Right), out var modelAction))
            {
                controller.SetGhostInput(
                    modelAction.horizontal,
                    modelAction.jump,
                    modelAction.drop,
                    modelAction.shove,
                    modelAction.routeName);
                return;
            }

            var target = GetTargetPoint(out var routeName);
            var direction = Mathf.Sign(target.x - transform.position.x);
            var horizontal = Mathf.Abs(target.x - transform.position.x) > 0.25f ? direction : 0f;
            var jump = ShouldJumpTowards(target);
            var drop = ShouldDropTowards(target);
            var shove = ShouldShove();
            controller.SetGhostInput(horizontal, jump, drop, shove, routeName);
        }

        private Vector2 GetTargetPoint(out string routeName)
        {
            if (controller.HasItem)
            {
                routeName = "Return";
                return manager.GetSpawnPoint(ArenaSide.Right);
            }

            var itemPosition = manager.Item.transform.position;
            var playerPosition = manager.Player.transform.position;
            var preferIntercept = playerPosition.x > -4f && itemPosition.y > 1.4f;

            if (itemPosition.y > 2.4f)
            {
                routeName = "Top";
                return preferIntercept ? new Vector2(3f, 3.2f) : itemPosition;
            }

            if (itemPosition.y < 1.4f)
            {
                routeName = "Bottom";
                return preferIntercept ? new Vector2(3f, -2.7f) : itemPosition;
            }

            routeName = "Center";
            return itemPosition;
        }

        private bool ShouldJumpTowards(Vector2 target)
        {
            if (!controller.IsGrounded())
            {
                return false;
            }

            if (target.y > transform.position.y + 0.75f)
            {
                return true;
            }

            var direction = Mathf.Sign(target.x - transform.position.x);
            var hit = Physics2D.CircleCast(transform.position, 0.2f, Vector2.right * direction, 0.55f);
            return hit.collider != null && hit.collider.gameObject != gameObject;
        }

        private bool ShouldDropTowards(Vector2 target)
        {
            return controller.IsGrounded() && target.y < transform.position.y - 0.8f;
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
