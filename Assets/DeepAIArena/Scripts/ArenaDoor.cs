using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(BoxCollider2D))]
    [RequireComponent(typeof(SpriteRenderer))]
    public class ArenaDoor : MonoBehaviour
    {
        [SerializeField] private float holdOpenSeconds = 2.5f;
        [SerializeField] private Color closedColor = new Color(0.28f, 0.32f, 0.42f, 1f);
        [SerializeField] private Color openColor = new Color(0.28f, 0.86f, 0.62f, 0.32f);

        private ArenaGameManager manager;
        private BoxCollider2D doorCollider;
        private SpriteRenderer spriteRenderer;
        private float openUntil = -999f;
        private bool wasOpen;
        private ArenaSide lastOpenedBy = ArenaSide.Left;
        private bool hasOpeningActor;

        public bool IsOpen => Time.time < openUntil;
        public float HoldOpenSeconds => holdOpenSeconds;

        public void Initialize(ArenaGameManager arenaGameManager)
        {
            manager = arenaGameManager;
            doorCollider = GetComponent<BoxCollider2D>();
            spriteRenderer = GetComponent<SpriteRenderer>();
            ApplyState(forceNotify: false);
        }

        public void RequestOpen(float duration, ArenaSide openedBy)
        {
            lastOpenedBy = openedBy;
            hasOpeningActor = true;
            OpenFor(duration);
        }

        public void RequestOpen(float duration)
        {
            hasOpeningActor = false;
            OpenFor(duration);
        }

        private void OpenFor(float duration)
        {
            openUntil = Mathf.Max(openUntil, Time.time + Mathf.Max(0.1f, duration));
            ApplyState(forceNotify: true);
        }

        public void ResetDoor()
        {
            openUntil = -999f;
            wasOpen = false;
            hasOpeningActor = false;
            ApplyState(forceNotify: false);
        }

        private void Awake()
        {
            doorCollider = GetComponent<BoxCollider2D>();
            spriteRenderer = GetComponent<SpriteRenderer>();
            ApplyState(forceNotify: false);
        }

        private void Update()
        {
            ApplyState(forceNotify: true);
        }

        private void ApplyState(bool forceNotify)
        {
            var open = IsOpen;
            if (doorCollider != null)
            {
                doorCollider.enabled = !open;
            }

            if (spriteRenderer != null)
            {
                spriteRenderer.color = open ? openColor : closedColor;
                spriteRenderer.sortingOrder = open ? -1 : 4;
            }

            if (forceNotify && open && !wasOpen)
            {
                manager?.ReportDoorOpened(this, hasOpeningActor ? lastOpenedBy : (ArenaSide?)null);
            }

            wasOpen = open;
        }
    }
}
