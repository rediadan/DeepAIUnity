using UnityEngine;

namespace DeepAIArena
{
    [RequireComponent(typeof(BoxCollider2D))]
    public class ArenaItem : MonoBehaviour
    {
        private ArenaCharacterController holder;
        private readonly Vector3 followOffset = new Vector3(0f, 0.45f, 0f);
        private ArenaCharacterController blockedCollector;
        private float pickupDisabledUntil;

        public bool IsHeld => holder != null;

        public void Initialize(ArenaGameManager arenaGameManager)
        {
        }

        public void AttachTo(ArenaCharacterController controller)
        {
            holder = controller;
        }

        public void Detach()
        {
            holder = null;
        }

        public void DropToWorld(ArenaCharacterController previousHolder, Vector3 position)
        {
            holder = null;
            blockedCollector = previousHolder;
            pickupDisabledUntil = Time.time + 0.35f;
            transform.position = position;
        }

        public bool CanBePickedUpBy(ArenaCharacterController controller)
        {
            if (Time.time >= pickupDisabledUntil)
            {
                blockedCollector = null;
            }

            return blockedCollector == null || blockedCollector != controller;
        }

        public void ResetToSpawn(Vector3 position)
        {
            holder = null;
            transform.position = position;
        }

        private void LateUpdate()
        {
            if (holder != null)
            {
                transform.position = holder.transform.position + followOffset;
            }
        }
    }
}
