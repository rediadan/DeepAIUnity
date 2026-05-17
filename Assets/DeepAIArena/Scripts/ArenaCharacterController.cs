using UnityEngine;
using UnityEngine.InputSystem;

namespace DeepAIArena
{
    [RequireComponent(typeof(Rigidbody2D))]
    [RequireComponent(typeof(CircleCollider2D))]
    public class ArenaCharacterController : MonoBehaviour
    {
        [SerializeField] private float moveSpeed = 6.5f;
        [SerializeField] private LayerMask obstacleMask = Physics2D.DefaultRaycastLayers;
        [SerializeField] private float wallCheckDistance = 0.12f;
        [SerializeField] private ArenaSide side;
        [SerializeField] private bool isGhost;
        [SerializeField] private float velocitySmoothing = 40f;
        [SerializeField] private float shoveRange = 0.75f;
        [SerializeField] private float shoveCooldown = 0.45f;
        [SerializeField] private float shoveForce = 4.5f;
        [SerializeField] private float knockbackControlLockDuration = 0.28f;

        private Rigidbody2D body;
        private CircleCollider2D circleCollider;
        private ArenaItem carriedItem;
        private ArenaGameManager arenaGameManager;
        private Vector2 moveInput;
        private Vector2 facingDirection = Vector2.right;
        private bool shoveRequested;
        private InputAction moveAction;
        private InputAction shoveAction;
        private float lastShoveTime = -999f;
        private float controlDisabledUntil = -999f;

        public ArenaSide Side
        {
            get => side;
            set => side = value;
        }

        public bool IsGhost
        {
            get => isGhost;
            set => isGhost = value;
        }

        public bool HasItem => carriedItem != null;
        public Transform GroundCheck { get; set; }
        public string DebugRouteName { get; set; } = "Center";
        public Rigidbody2D Body => body;

        public void Initialize(ArenaGameManager arenaGameManager)
        {
            body = GetComponent<Rigidbody2D>();
            circleCollider = GetComponent<CircleCollider2D>();
            this.arenaGameManager = arenaGameManager;
            body.gravityScale = 0f;
            body.freezeRotation = true;
        }

        private void OnEnable()
        {
            if (IsGhost)
            {
                return;
            }

            moveAction ??= new InputAction(
                name: "Move",
                type: InputActionType.Value,
                binding: "<Gamepad>/leftStick");
            moveAction.AddCompositeBinding("2DVector")
                .With("Up", "<Keyboard>/w")
                .With("Up", "<Keyboard>/upArrow")
                .With("Down", "<Keyboard>/s")
                .With("Down", "<Keyboard>/downArrow")
                .With("Left", "<Keyboard>/a")
                .With("Left", "<Keyboard>/leftArrow")
                .With("Right", "<Keyboard>/d")
                .With("Right", "<Keyboard>/rightArrow");

            shoveAction ??= new InputAction(
                name: "Shove",
                type: InputActionType.Button,
                binding: "<Keyboard>/leftCtrl");
            shoveAction.AddBinding("<Keyboard>/rightCtrl");
            shoveAction.AddBinding("<Keyboard>/e");
            shoveAction.AddBinding("<Gamepad>/rightShoulder");

            moveAction.Enable();
            shoveAction.Enable();
        }

        private void OnDisable()
        {
            moveAction?.Disable();
            shoveAction?.Disable();
        }

        private void Update()
        {
            if (IsGhost)
            {
                return;
            }

            moveInput = moveAction != null ? Vector2.ClampMagnitude(moveAction.ReadValue<Vector2>(), 1f) : Vector2.zero;
            UpdateFacingDirection(moveInput);
            if (shoveAction != null && shoveAction.WasPressedThisFrame())
            {
                shoveRequested = true;
            }
        }

        private void FixedUpdate()
        {
            if (body == null)
            {
                return;
            }

            if (Time.time >= controlDisabledUntil)
            {
                var desiredInput = IsBlockedByWall(moveInput) ? Vector2.zero : moveInput;
                var targetVelocity = desiredInput * moveSpeed;
                body.linearVelocity = Vector2.MoveTowards(
                    body.linearVelocity,
                    targetVelocity,
                    velocitySmoothing * Time.fixedDeltaTime);
            }

            if (shoveRequested)
            {
                TryShove();
            }

            shoveRequested = false;
        }

        public void SetGhostInput(Vector2 move, bool shove, string routeName)
        {
            if (!IsGhost)
            {
                return;
            }

            moveInput = Vector2.ClampMagnitude(move, 1f);
            UpdateFacingDirection(moveInput);
            DebugRouteName = routeName;
            if (shove)
            {
                shoveRequested = true;
            }
        }

        public bool IsGrounded()
        {
            return true;
        }

        public bool HasGroundBelow()
        {
            return true;
        }

        public bool CanDropDown()
        {
            return false;
        }

        public bool IsWallAhead()
        {
            return IsBlockedByWall(facingDirection);
        }

        public bool IsBlockedByWall(Vector2 inputDirection)
        {
            if (inputDirection.sqrMagnitude < 0.0001f)
            {
                return false;
            }

            circleCollider ??= GetComponent<CircleCollider2D>();
            if (circleCollider == null)
            {
                return false;
            }

            var direction = inputDirection.normalized;
            var radius = Mathf.Max(circleCollider.radius * transform.lossyScale.x * 0.65f, 0.08f);
            var hit = Physics2D.CircleCast(transform.position, radius, direction, wallCheckDistance, obstacleMask);
            return hit.collider != null && !hit.transform.IsChildOf(transform) && !hit.collider.isTrigger;
        }

        private void TryShove()
        {
            if (Time.time - lastShoveTime < shoveCooldown)
            {
                return;
            }

            var hits = Physics2D.OverlapCircleAll(transform.position, shoveRange);
            ArenaCharacterController other = null;
            var nearestDistance = float.MaxValue;
            foreach (var hit in hits)
            {
                if (!hit.TryGetComponent(out ArenaCharacterController candidate) || candidate == this || candidate.Side == Side)
                {
                    continue;
                }

                var distance = Vector2.Distance(transform.position, candidate.transform.position);
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    other = candidate;
                }
            }

            if (other == null)
            {
                arenaGameManager?.ReportShoveAttempt(Side, false, false);
                lastShoveTime = Time.time;
                return;
            }

            var forceDirection = ((Vector2)(other.transform.position - transform.position)).normalized;
            var forcedItemDrop = other.HasItem;
            arenaGameManager?.ReportShoveAttempt(Side, true, forcedItemDrop);
            other.ApplyKnockback(forceDirection, shoveForce, knockbackControlLockDuration);

            if (other.HasItem)
            {
                other.ForceDropItem();
                arenaGameManager?.ReportReward(ArenaRewardEventType.ItemLost, other.Side, -2f, "item_shoved_away");
                arenaGameManager?.ReportReward(ArenaRewardEventType.ItemLost, Side, 2f, "forced_drop");
            }

            lastShoveTime = Time.time;
        }

        public void ApplyKnockback(Vector2 direction, float force, float controlLockDuration)
        {
            if (direction.sqrMagnitude < 0.0001f)
            {
                direction = facingDirection.sqrMagnitude > 0.0001f ? facingDirection : Vector2.right;
            }

            body ??= GetComponent<Rigidbody2D>();
            if (body == null)
            {
                return;
            }

            controlDisabledUntil = Time.time + Mathf.Max(0f, controlLockDuration);
            body.linearVelocity = direction.normalized * force;
            moveInput = Vector2.zero;
            shoveRequested = false;
        }

        private void OnDrawGizmosSelected()
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, shoveRange);

            Gizmos.color = IsWallAhead() ? Color.red : Color.green;
            Gizmos.DrawLine(transform.position, (Vector2)transform.position + facingDirection.normalized * wallCheckDistance);
        }

        private void UpdateFacingDirection(Vector2 input)
        {
            if (input.sqrMagnitude > 0.0001f)
            {
                facingDirection = input.normalized;
            }
        }

        public void PickupItem(ArenaItem item)
        {
            carriedItem = item;
            item.AttachTo(this);
            arenaGameManager?.ReportReward(ArenaRewardEventType.ItemCollected, Side, 1f, "item_pickup");
        }

        public void DropItem()
        {
            if (carriedItem == null)
            {
                return;
            }

            carriedItem.Detach();
            carriedItem = null;
        }

        public ArenaActionSnapshot GetCurrentActionSnapshot()
        {
            return new ArenaActionSnapshot
            {
                moveAction = ToMoveAction(moveInput),
                shovePressed = shoveRequested,
                routeName = DebugRouteName
            };
        }

        public void ForceDropItem()
        {
            if (carriedItem == null)
            {
                return;
            }

            var dropDirection = moveInput.sqrMagnitude > 0.0001f ? moveInput.normalized : facingDirection.normalized;
            carriedItem.DropToWorld(this, transform.position + (Vector3)(dropDirection * 0.45f));
            carriedItem = null;
        }

        public void ResetActor(Vector3 position)
        {
            transform.position = position;
            if (body == null)
            {
                body = GetComponent<Rigidbody2D>();
            }

            body.gravityScale = 0f;
            body.linearVelocity = Vector2.zero;
            moveInput = Vector2.zero;
            shoveRequested = false;
            controlDisabledUntil = -999f;
            carriedItem = null;
        }

        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.TryGetComponent(out ArenaItem arenaItem) && !HasItem && !arenaItem.IsHeld && arenaItem.CanBePickedUpBy(this))
            {
                PickupItem(arenaItem);
            }
        }

        private static ArenaMoveAction ToMoveAction(Vector2 input)
        {
            if (input.sqrMagnitude < 0.01f)
            {
                return ArenaMoveAction.Idle;
            }

            if (Mathf.Abs(input.x) >= Mathf.Abs(input.y))
            {
                return input.x < 0f ? ArenaMoveAction.MoveLeft : ArenaMoveAction.MoveRight;
            }

            return input.y < 0f ? ArenaMoveAction.MoveDown : ArenaMoveAction.MoveUp;
        }
    }
}
