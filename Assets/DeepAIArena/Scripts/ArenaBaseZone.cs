using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(BoxCollider2D))]
    public class ArenaBaseZone : MonoBehaviour
    {
        private ArenaGameManager manager;

        [SerializeField] private ArenaSide side;

        public ArenaSide Side
        {
            get => side;
            set => side = value;
        }

        public void Initialize(ArenaGameManager arenaGameManager)
        {
            manager = arenaGameManager;
        }

        private void OnTriggerEnter2D(Collider2D other)
        {
            if (manager == null || !other.TryGetComponent(out ArenaCharacterController controller))
            {
                return;
            }

            if (controller.Side == Side && controller.HasItem)
            {
                manager.Deliver(controller);
            }
        }
    }
}
