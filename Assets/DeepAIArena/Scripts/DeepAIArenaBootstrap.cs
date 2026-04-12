using UnityEngine;

namespace DeepAIArena
{
    public class DeepAIArenaBootstrap : MonoBehaviour
    {
        [SerializeField] private bool buildOnPlay = true;

        private void Awake()
        {
            if (!Application.isPlaying || !buildOnPlay)
            {
                return;
            }

            if (FindAnyObjectByType<ArenaGameManager>() != null)
            {
                return;
            }

            ArenaBuilder.Build(transform);
        }
    }
}
