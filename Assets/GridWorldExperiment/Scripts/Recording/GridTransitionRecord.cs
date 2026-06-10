using System;

namespace GridWorldExperiment
{
    [Serializable]
    public class GridTransitionRecord
    {
        public int episode;
        public int step;
        public float[] state;
        public int action;
        public float reward;
        public float[] next_state;
        public bool done;
        public string eventName;
    }
}
