using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(BoxCollider2D))]
    public class ArenaBaseZone : MonoBehaviour
    {
        private ArenaGameManager manager;

        [SerializeField] private ArenaSide side;
        [SerializeField] private bool sharedBase;

        public ArenaSide Side
        {
            get => side;
            set => side = value;
        }

        public bool SharedBase
        {
            get => sharedBase;
            set => sharedBase = value;
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

            if (controller.HasItem && (sharedBase || controller.Side == Side))
            {
                manager.Deliver(controller);
            }
        }
    }
}
