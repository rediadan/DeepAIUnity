using UnityEngine;

namespace GridWorldExperiment
{
    public class RewardCalculator : MonoBehaviour
    {
        [SerializeField] private float coinReward = 1f;
        [SerializeField] private float deathReward = -1f;
        [SerializeField] private float trapReward = -0.5f;
        [SerializeField] private float wallHitReward = -0.1f;
        [SerializeField] private float stepReward = -0.01f;
        [SerializeField] private float clearBonusReward = 2f;

        public float StepReward => stepReward;
        public float WallHitReward => wallHitReward;
        public float TrapReward => trapReward;
        public float CoinReward => coinReward;
        public float DeathReward => deathReward;
        public float ClearBonusReward => clearBonusReward;
    }
}
