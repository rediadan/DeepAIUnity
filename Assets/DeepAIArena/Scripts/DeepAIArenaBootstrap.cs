using UnityEngine;

namespace DeepAIArena
{
    public class DeepAIArenaBootstrap : MonoBehaviour
    {
        [SerializeField] private bool buildOnPlay = true;

        private void Awake()
        {
            if (!Application.isPlaying)
            {
                return;
            }

            var manager = FindAnyObjectByType<ArenaGameManager>();
            if (manager != null)
            {
                var arenaRoot = manager.transform.parent != null ? manager.transform.parent : transform;
                ArenaBuilder.EnsureV2Environment(arenaRoot);
                manager.RefreshEnvironmentReferences();
                return;
            }

            if (!buildOnPlay)
            {
                return;
            }

            ArenaBuilder.Build(transform);
        }
    }
}
