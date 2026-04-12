using UnityEngine;

namespace DeepAIArena
{
    public enum PlatformTraversalMode
    {
        Both,
        UpOnly,
        DownOnly,
        Neither
    }

    [ExecuteAlways]
    [RequireComponent(typeof(BoxCollider2D))]
    public class ArenaPlatform : MonoBehaviour
    {
        [SerializeField] private PlatformTraversalMode traversalMode = PlatformTraversalMode.Both;

        public PlatformTraversalMode TraversalMode => traversalMode;
        public bool AllowPassFromBelow => traversalMode == PlatformTraversalMode.Both || traversalMode == PlatformTraversalMode.UpOnly;
        public bool AllowDropFromAbove => traversalMode == PlatformTraversalMode.Both || traversalMode == PlatformTraversalMode.DownOnly;

        public void SetTraversalMode(PlatformTraversalMode mode)
        {
            traversalMode = mode;
        }

        public void ApplySettings(BoxCollider2D collider = null)
        {
            collider ??= GetComponent<BoxCollider2D>();
            if (collider == null)
            {
                return;
            }

            var effector = GetComponent<PlatformEffector2D>();
            if (AllowPassFromBelow)
            {
                if (effector == null)
                {
                    effector = gameObject.AddComponent<PlatformEffector2D>();
                }

                effector.useOneWay = true;
                effector.surfaceArc = 170f;
                effector.useColliderMask = false;
                collider.usedByEffector = true;
            }
            else
            {
                if (effector != null)
                {
                    if (Application.isPlaying)
                    {
                        Destroy(effector);
                    }
                    else
                    {
                        DestroyImmediate(effector);
                    }
                }

                collider.usedByEffector = false;
            }
        }

        private void OnValidate()
        {
            ApplySettings();
        }
    }
}
