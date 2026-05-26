using System.Collections.Generic;
using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(BoxCollider2D))]
    [RequireComponent(typeof(SpriteRenderer))]
    public class ArenaSwitch : MonoBehaviour
    {
        [SerializeField] private ArenaDoor linkedDoor;
        [SerializeField] private float doorOpenSeconds = 2.5f;
        [SerializeField] private Color inactiveColor = new Color(0.2f, 0.48f, 0.85f, 1f);
        [SerializeField] private Color activeColor = new Color(0.45f, 1f, 0.66f, 1f);

        private readonly HashSet<ArenaCharacterController> actorsOnSwitch = new();
        private ArenaGameManager manager;
        private SpriteRenderer spriteRenderer;
        private bool wasActive;

        public bool IsActive => actorsOnSwitch.Count > 0;
        public ArenaDoor LinkedDoor => linkedDoor;

        public void Initialize(ArenaGameManager arenaGameManager)
        {
            manager = arenaGameManager;
            spriteRenderer = GetComponent<SpriteRenderer>();
            ApplyState();
        }

        public void SetLinkedDoor(ArenaDoor door)
        {
            linkedDoor = door;
        }

        public void SetDoorOpenSeconds(float seconds)
        {
            doorOpenSeconds = Mathf.Max(0.1f, seconds);
        }

        public void ResetSwitch()
        {
            actorsOnSwitch.Clear();
            wasActive = false;
            ApplyState();
        }

        private void Awake()
        {
            var switchCollider = GetComponent<BoxCollider2D>();
            switchCollider.isTrigger = true;
            spriteRenderer = GetComponent<SpriteRenderer>();
            ApplyState();
        }

        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.TryGetComponent(out ArenaCharacterController controller))
            {
                actorsOnSwitch.Add(controller);
                ActivateLinkedDoorIfNeeded();
            }
        }

        private void OnTriggerStay2D(Collider2D other)
        {
            if (other.TryGetComponent(out ArenaCharacterController controller))
            {
                actorsOnSwitch.Add(controller);
                ActivateLinkedDoorIfNeeded();
            }
        }

        private void OnTriggerExit2D(Collider2D other)
        {
            if (other.TryGetComponent(out ArenaCharacterController controller))
            {
                actorsOnSwitch.Remove(controller);
                ApplyState();
            }
        }

        private void ActivateLinkedDoorIfNeeded()
        {
            if (!IsActive)
            {
                ApplyState();
                return;
            }

            var notifiedDoor = false;
            if (!wasActive)
            {
                foreach (var actor in actorsOnSwitch)
                {
                    if (actor != null)
                    {
                        if (!notifiedDoor)
                        {
                            linkedDoor?.RequestOpen(doorOpenSeconds, actor.Side);
                            notifiedDoor = true;
                        }

                        manager?.ReportSwitchActivated(this, actor.Side);
                    }
                }
            }

            if (!notifiedDoor)
            {
                linkedDoor?.RequestOpen(doorOpenSeconds);
            }

            ApplyState();
        }

        private void ApplyState()
        {
            var active = IsActive;
            if (spriteRenderer != null)
            {
                spriteRenderer.color = active ? activeColor : inactiveColor;
                spriteRenderer.sortingOrder = 3;
            }

            wasActive = active;
        }
    }
}
