namespace GridWorldExperiment
{
    public struct GridStepResult
    {
        public float reward;
        public bool done;
        public string eventName;
        public int score;
        public int steps;
        public bool hitWall;
        public bool hitTrap;
        public bool collectedCoin;
        public bool death;
    }
}
