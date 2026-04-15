using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections;

namespace DeepAIArena
{
    [RequireComponent(typeof(Rigidbody2D))]
    [RequireComponent(typeof(CircleCollider2D))]
    public class ArenaCharacterController : MonoBehaviour
    {
        [SerializeField] private float moveSpeed = 8f;
        [SerializeField] private float jumpForce = 10f;
        [SerializeField] private LayerMask groundMask = Physics2D.DefaultRaycastLayers;
        [SerializeField] private float groundCheckDistance = 0.1f;
        [SerializeField] private float groundCheckOriginOffset = -0.01f;
        [SerializeField] private float wallCheckDistance = 0.08f;
        [SerializeField] private float groundBelowCheckDistance = 0.7f;
        [SerializeField] private float platformDropDistance = 0.2f;
        [SerializeField] private float platformDropDuration = 0.3f;
        [SerializeField] private ArenaSide side;
        [SerializeField] private bool isGhost;
        [SerializeField] private float shoveRange = 0.65f;
        [SerializeField] private float shoveCooldown = 0.45f;
        [SerializeField] private float shoveForce = 4.5f;

        private Rigidbody2D body;
        private CircleCollider2D circleCollider;
        private ArenaItem carriedItem;
        private ArenaGameManager arenaGameManager;
        private float moveInput;
        private bool jumpRequested;
        private bool dropRequested;
        private bool shoveRequested;
        private InputAction moveAction;
        private InputAction jumpAction;
        private InputAction downAction;
        private InputAction shoveAction;
        private Coroutine dropCoroutine;
        private Collider2D ignoredPlatform;
        private float lastShoveTime = -999f;
        private float facingDirection = 1f;

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
                binding: "<Gamepad>/leftStick/x");
            moveAction.AddCompositeBinding("1DAxis")
                .With("Negative", "<Keyboard>/a")
                .With("Negative", "<Keyboard>/leftArrow")
                .With("Positive", "<Keyboard>/d")
                .With("Positive", "<Keyboard>/rightArrow");

            jumpAction ??= new InputAction(
                name: "Jump",
                type: InputActionType.Button,
                binding: "<Keyboard>/space");
            jumpAction.AddBinding("<Keyboard>/w");
            jumpAction.AddBinding("<Keyboard>/upArrow");
            jumpAction.AddBinding("<Gamepad>/buttonSouth");

            downAction ??= new InputAction(
                name: "Drop",
                type: InputActionType.Button,
                binding: "<Keyboard>/s");
            downAction.AddBinding("<Keyboard>/downArrow");
            downAction.AddBinding("<Gamepad>/dpad/down");
            downAction.AddBinding("<Gamepad>/leftStick/down");

            shoveAction ??= new InputAction(
                name: "Shove",
                type: InputActionType.Button,
                binding: "<Keyboard>/leftCtrl");
            shoveAction.AddBinding("<Keyboard>/rightCtrl");
            shoveAction.AddBinding("<Keyboard>/e");
            shoveAction.AddBinding("<Gamepad>/rightShoulder");

            moveAction.Enable();
            jumpAction.Enable();
            downAction.Enable();
            shoveAction.Enable();
        }

        private void OnDisable()
        {
            moveAction?.Disable();
            jumpAction?.Disable();
            downAction?.Disable();
            shoveAction?.Disable();
        }

        private void Update()
        {
            if (IsGhost)
            {
                return;
            }

            moveInput = moveAction != null ? moveAction.ReadValue<float>() : 0f;
            UpdateFacingDirection(moveInput);
            if (jumpAction != null && jumpAction.WasPressedThisFrame())
            {
                jumpRequested = true;
            }

            if (downAction != null && downAction.WasPressedThisFrame())
            {
                dropRequested = true;
            }

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

            var horizontalVelocity = moveInput * moveSpeed;
            if (IsBlockedByWall(moveInput))
            {
                horizontalVelocity = 0f;
            }

            body.linearVelocity = new Vector2(horizontalVelocity, body.linearVelocity.y);

            if (jumpRequested && IsGrounded())
            {
                body.linearVelocity = new Vector2(body.linearVelocity.x, jumpForce);
            }

            if (dropRequested)
            {
                TryDropThroughPlatform();
            }

            if (shoveRequested)
            {
                TryShove();
            }

            jumpRequested = false;
            dropRequested = false;
            shoveRequested = false;
        }

        public void SetGhostInput(float horizontal, bool jump, bool drop, bool shove, string routeName)
        {
            if (!IsGhost)
            {
                return;
            }

            moveInput = Mathf.Clamp(horizontal, -1f, 1f);
            UpdateFacingDirection(moveInput);
            DebugRouteName = routeName;

            if (jump)
            {
                jumpRequested = true;
            }

            if (drop)
            {
                dropRequested = true;
            }

            if (shove)
            {
                shoveRequested = true;
            }
        }

        public bool IsGrounded()
        {
            circleCollider ??= GetComponent<CircleCollider2D>();
            if (circleCollider == null)
            {
                return false;
            }

            var bounds = circleCollider.bounds;
            var origin = new Vector2(bounds.center.x, bounds.min.y + groundCheckOriginOffset);
            var hit = Physics2D.Raycast(origin, Vector2.down, groundCheckDistance, groundMask);
            return hit.collider != null
                && hit.collider != ignoredPlatform
                && !hit.transform.IsChildOf(transform)
                && hit.distance <= groundCheckDistance;
        }

        public bool HasGroundBelow()
        {
            circleCollider ??= GetComponent<CircleCollider2D>();
            if (circleCollider == null)
            {
                return false;
            }

            var bounds = circleCollider.bounds;
            var origin = new Vector2(bounds.center.x, bounds.min.y + groundCheckOriginOffset);
            var hit = Physics2D.Raycast(origin, Vector2.down, groundBelowCheckDistance, groundMask);
            return hit.collider != null
                && hit.collider != ignoredPlatform
                && !hit.transform.IsChildOf(transform);
        }

        public bool CanDropDown()
        {
            circleCollider ??= GetComponent<CircleCollider2D>();
            if (circleCollider == null || !IsGrounded())
            {
                return false;
            }

            var bounds = circleCollider.bounds;
            var origin = new Vector2(bounds.center.x, bounds.min.y + groundCheckOriginOffset);
            var hit = Physics2D.Raycast(origin, Vector2.down, platformDropDistance, groundMask);
            return hit.collider != null
                && !hit.transform.IsChildOf(transform)
                && hit.collider.TryGetComponent(out ArenaPlatform platform)
                && platform.AllowDropFromAbove;
        }

        public bool IsWallAhead()
        {
            return IsBlockedByWall(facingDirection);
        }

        private bool IsBlockedByWall(float horizontalInput)
        {
            if (Mathf.Abs(horizontalInput) < 0.01f)
            {
                return false;
            }

            circleCollider ??= GetComponent<CircleCollider2D>();
            if (circleCollider == null)
            {
                return false;
            }

            var bounds = circleCollider.bounds;
            var origin = bounds.center;
            var radius = Mathf.Max(bounds.extents.y * 0.45f, 0.05f);
            var direction = horizontalInput > 0f ? Vector2.right : Vector2.left;
            var hit = Physics2D.CircleCast(origin, radius, direction, wallCheckDistance, groundMask);

            return hit.collider != null
                && hit.collider != ignoredPlatform
                && !hit.transform.IsChildOf(transform);
        }

        private void TryDropThroughPlatform()
        {
            if (!IsGrounded() || circleCollider == null || dropCoroutine != null)
            {
                return;
            }

            var bounds = circleCollider.bounds;
            var origin = new Vector2(bounds.center.x, bounds.min.y + groundCheckOriginOffset);
            var hit = Physics2D.Raycast(origin, Vector2.down, platformDropDistance, groundMask);
            if (hit.collider == null || hit.transform.IsChildOf(transform))
            {
                return;
            }

            if (!hit.collider.TryGetComponent(out ArenaPlatform platform) || !platform.AllowDropFromAbove)
            {
                return;
            }

            dropCoroutine = StartCoroutine(DropThroughPlatformRoutine(hit.collider));
        }

        private IEnumerator DropThroughPlatformRoutine(Collider2D platformCollider)
        {
            ignoredPlatform = platformCollider;
            Physics2D.IgnoreCollision(circleCollider, platformCollider, true);
            body.linearVelocity = new Vector2(body.linearVelocity.x, Mathf.Min(body.linearVelocity.y, -2f));

            yield return new WaitForSeconds(platformDropDuration);

            Physics2D.IgnoreCollision(circleCollider, platformCollider, false);
            ignoredPlatform = null;
            dropCoroutine = null;
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
                lastShoveTime = Time.time;
                return;
            }

            var forceDirection = ((Vector2)(other.transform.position - transform.position)).normalized;
            if (other.Body != null)
            {
                other.Body.AddForce(forceDirection * shoveForce, ForceMode2D.Impulse);
            }

            if (other.HasItem)
            {
                other.ForceDropItem();
                arenaGameManager?.ReportReward(ArenaRewardEventType.ItemLost, other.Side, -2f, "item_shoved_away");
                arenaGameManager?.ReportReward(ArenaRewardEventType.ItemLost, Side, 2f, "forced_drop");
            }

            lastShoveTime = Time.time;
        }

        private void OnDrawGizmosSelected()
        {
            var activeCollider = circleCollider != null ? circleCollider : GetComponent<CircleCollider2D>();
            if (activeCollider == null)
            {
                return;
            }

            var bounds = activeCollider.bounds;
            var origin = new Vector3(bounds.center.x, bounds.min.y + groundCheckOriginOffset, transform.position.z);
            var end = origin + Vector3.down * groundCheckDistance;

            Gizmos.color = IsGrounded() ? Color.green : Color.red;
            Gizmos.DrawSphere(origin, 0.03f);
            Gizmos.DrawLine(origin, end);
            Gizmos.DrawSphere(end, 0.025f);

            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, shoveRange);
        }

        private void UpdateFacingDirection(float horizontalInput)
        {
            if (horizontalInput > 0.01f)
            {
                facingDirection = 1f;
            }
            else if (horizontalInput < -0.01f)
            {
                facingDirection = -1f;
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
            var moveActionSnapshot = ArenaMoveAction.Idle;
            if (moveInput < -0.1f)
            {
                moveActionSnapshot = ArenaMoveAction.MoveLeft;
            }
            else if (moveInput > 0.1f)
            {
                moveActionSnapshot = ArenaMoveAction.MoveRight;
            }

            return new ArenaActionSnapshot
            {
                moveAction = moveActionSnapshot,
                jumpPressed = jumpRequested,
                dropPressed = dropRequested,
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

            var dropDirection = moveInput < -0.1f ? Vector3.left : Vector3.right;
            carriedItem.DropToWorld(this, transform.position + Vector3.up * 0.25f + dropDirection * 0.35f);
            carriedItem = null;
        }

        public void ResetActor(Vector3 position)
        {
            transform.position = position;
            if (body == null)
            {
                body = GetComponent<Rigidbody2D>();
            }

            body.linearVelocity = Vector2.zero;
            moveInput = 0f;
            jumpRequested = false;
            carriedItem = null;
        }

        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.TryGetComponent(out ArenaItem arenaItem) && !HasItem && !arenaItem.IsHeld && arenaItem.CanBePickedUpBy(this))
            {
                PickupItem(arenaItem);
            }
        }
    }
}
