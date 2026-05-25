using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(BoxCollider2D))]
    [RequireComponent(typeof(Rigidbody2D))]
    [RequireComponent(typeof(SpriteRenderer))]
    public class ArenaMovingObstacle : MonoBehaviour
    {
        [SerializeField] private Vector2 localPointA = new Vector2(-1f, 0f);
        [SerializeField] private Vector2 localPointB = new Vector2(1f, 0f);
        [SerializeField] private float speed = 1.8f;

        private Rigidbody2D body;
        private Vector2 origin;
        private Vector2 previousPosition;
        private float pathTime;
        private bool movingToB = true;

        public Vector2 Velocity { get; private set; }

        public void Configure(Vector2 pointA, Vector2 pointB, float moveSpeed)
        {
            localPointA = pointA;
            localPointB = pointB;
            speed = Mathf.Max(0.1f, moveSpeed);
        }

        public void ResetMotion()
        {
            origin = transform.parent != null ? (Vector2)transform.parent.position : Vector2.zero;
            pathTime = 0f;
            movingToB = true;
            var start = origin + localPointA;
            transform.position = start;
            previousPosition = start;
            Velocity = Vector2.zero;
            if (body != null)
            {
                body.position = start;
                body.linearVelocity = Vector2.zero;
            }
        }

        private void Awake()
        {
            body = GetComponent<Rigidbody2D>();
            body.gravityScale = 0f;
            body.freezeRotation = true;
            body.bodyType = RigidbodyType2D.Kinematic;
            ResetMotion();
        }

        private void FixedUpdate()
        {
            body ??= GetComponent<Rigidbody2D>();
            var pointA = origin + localPointA;
            var pointB = origin + localPointB;
            var from = movingToB ? pointA : pointB;
            var to = movingToB ? pointB : pointA;
            var distance = Vector2.Distance(pointA, pointB);
            var duration = Mathf.Max(distance / Mathf.Max(speed, 0.1f), 0.1f);

            pathTime += Time.fixedDeltaTime;
            var t = Mathf.Clamp01(pathTime / duration);
            var nextPosition = Vector2.Lerp(from, to, t);
            Velocity = (nextPosition - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.0001f);
            previousPosition = nextPosition;
            body.MovePosition(nextPosition);

            if (t >= 1f)
            {
                movingToB = !movingToB;
                pathTime = 0f;
            }
        }
    }
}
