using UnityEngine;

namespace DeepAIArena
{
    public class DeepAIArenaBootstrap : MonoBehaviour
    {
        [SerializeField] private bool buildOnPlay = true;
        [SerializeField] private bool overrideActorModesOnPlay;
        [SerializeField] private ArenaActorControlMode leftActorModeOnPlay = ArenaActorControlMode.RuleBased;
        [SerializeField] private ArenaActorControlMode rightActorModeOnPlay = ArenaActorControlMode.MlAgents;

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
                ApplyActorModeOverride(manager);
                return;
            }

            if (!buildOnPlay)
            {
                return;
            }

            ArenaBuilder.Build(transform);
            ApplyActorModeOverride(FindAnyObjectByType<ArenaGameManager>());
        }

        private void ApplyActorModeOverride(ArenaGameManager manager)
        {
            if (!overrideActorModesOnPlay || manager == null)
            {
                return;
            }

            manager.SetActorControlModes(leftActorModeOnPlay, rightActorModeOnPlay);
        }
    }
}
