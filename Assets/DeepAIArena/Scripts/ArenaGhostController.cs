using UnityEngine;

namespace DeepAIArena
{
    public enum ArenaGhostPolicyMode
    {
        RuleBased,
        OnnxInference,
        DqnInference,
        RasterDqnInference
    }

    public enum ArenaGhostRuntimeMode
    {
        RuleBased,
        OnnxInference,
        DqnInference,
        RasterDqnInference,
        OnnxFailed,
        DqnFailed,
        RasterDqnFailed,
        OnnxFallback
    }

    [RequireComponent(typeof(ArenaCharacterController))]
    public class ArenaGhostController : MonoBehaviour
    {
        private const float LeftStartDividerX = -6.78f;
        private const float RightStartDividerX = 6.78f;
        private const float StartDividerLowerEdgeY = 0.87f;
        private const float StartDividerUpperEdgeY = 3.47f;
        private const float StartDividerBypassY = 0.2f;
        private const float DoorDetourX = 1.9f;
        private const float TopLaneY = 3.25f;
        private const float BottomLaneY = -2.3f;
        private const float CenterLaneY = 0f;

        [Header("Ghost Policy")]
        [Tooltip("Rule-Based keeps handcrafted logic. ONNX/DQN Inference use vector features. Raster DQN Inference uses CNN raster input.")]
        [SerializeField] private ArenaGhostPolicyMode policyMode = ArenaGhostPolicyMode.RuleBased;
        [SerializeField] private ArenaGhostOnnxPolicy onnxPolicy;
        [SerializeField] private ArenaRasterOnnxPolicy rasterOnnxPolicy;
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
            rasterOnnxPolicy ??= GetComponent<ArenaRasterOnnxPolicy>();
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
            else if (policyMode == ArenaGhostPolicyMode.RasterDqnInference)
            {
                rasterOnnxPolicy ??= GetComponent<ArenaRasterOnnxPolicy>();
                if (rasterOnnxPolicy != null
                    && rasterOnnxPolicy.TryEvaluate(out var modelAction))
                {
                    RuntimeMode = ArenaGhostRuntimeMode.RasterDqnInference;
                    controller.SetGhostInput(
                        modelAction.move,
                        modelAction.shove,
                        modelAction.routeName);
                    return;
                }

                if (!allowRuleFallbackOnInferenceFailure)
                {
                    RuntimeMode = ArenaGhostRuntimeMode.RasterDqnFailed;
                    controller.SetGhostInput(Vector2.zero, false, "RasterDqnFailed");
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
            target = ApplyMapWaypoints(observation, target, ref routeName);
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
            if (ShouldOpenDoorBeforeShortcut(observation))
            {
                routeName = "Switch";
                return SelectSwitchPoint(observation);
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

        private Vector2 ApplyMapWaypoints(ArenaObservationSnapshot observation, Vector2 target, ref string routeName)
        {
            var position = observation.selfPosition;
            if (NeedsStartDividerBypass(position, target))
            {
                routeName = "StartBypass";
                return new Vector2(position.x < 0f ? LeftStartDividerX - 0.75f : RightStartDividerX + 0.75f, StartDividerBypassY);
            }

            if (observation.detourNeeded && !observation.doorOpen)
            {
                routeName = "DoorDetour";
                return SelectDoorDetourPoint(position, target);
            }

            if (routeName == "Top" && Mathf.Abs(position.y - TopLaneY) > 0.35f && Mathf.Abs(position.x) > DoorDetourX)
            {
                routeName = "TopLaneAlign";
                return new Vector2(position.x, TopLaneY);
            }

            if ((routeName == "Bottom" || routeName == "Escape") && Mathf.Abs(position.y - BottomLaneY) > 0.35f && Mathf.Abs(position.x) > DoorDetourX)
            {
                routeName = "BottomLaneAlign";
                return new Vector2(position.x, BottomLaneY);
            }

            return target;
        }

        private static bool ShouldOpenDoorBeforeShortcut(ArenaObservationSnapshot observation)
        {
            return !observation.doorOpen
                && !observation.switchActive
                && observation.shortcutBlocked
                && observation.switchDelta.sqrMagnitude <= 36f;
        }

        private static Vector2 SelectSwitchPoint(ArenaObservationSnapshot observation)
        {
            return observation.selfPosition + observation.switchDelta;
        }

        private static bool NeedsStartDividerBypass(Vector2 position, Vector2 target)
        {
            if (position.x < LeftStartDividerX
                && target.x > LeftStartDividerX
                && position.y > StartDividerLowerEdgeY
                && position.y < StartDividerUpperEdgeY)
            {
                return true;
            }

            return position.x > RightStartDividerX
                && target.x < RightStartDividerX
                && position.y > StartDividerLowerEdgeY
                && position.y < StartDividerUpperEdgeY;
        }

        private static Vector2 SelectDoorDetourPoint(Vector2 position, Vector2 target)
        {
            var detourY = target.y >= CenterLaneY ? TopLaneY : BottomLaneY;
            if (Mathf.Abs(position.x) < DoorDetourX)
            {
                detourY = position.y >= CenterLaneY ? TopLaneY : BottomLaneY;
            }

            var detourX = position.x < 0f ? -DoorDetourX : DoorDetourX;
            if (Mathf.Abs(position.y - detourY) > 0.45f)
            {
                return new Vector2(position.x, detourY);
            }

            return new Vector2(detourX, detourY);
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
