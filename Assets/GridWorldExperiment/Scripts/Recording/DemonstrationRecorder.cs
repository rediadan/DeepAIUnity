using System.IO;
using System.Text;
using UnityEngine;

namespace GridWorldExperiment
{
    public class DemonstrationRecorder : MonoBehaviour
    {
        [SerializeField] private GridWorldManager manager;
        [SerializeField] private StateEncoder stateEncoder;
        [SerializeField] private string outputRelativePath = "GridWorldExperiment/datasets/demonstrations/unity_demo_raw.jsonl";
        [SerializeField] private bool recordingEnabled = true;

        private int episodeIndex;
        private string outputPath;

        private void Reset()
        {
            manager = GetComponent<GridWorldManager>();
            stateEncoder = GetComponent<StateEncoder>();
        }

        private void Awake()
        {
            if (manager == null)
            {
                manager = GetComponent<GridWorldManager>();
            }

            if (stateEncoder == null)
            {
                stateEncoder = GetComponent<StateEncoder>();
            }

            outputPath = Path.Combine(Application.dataPath, "..", outputRelativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
        }

        public void BeginNewEpisode()
        {
            episodeIndex++;
        }

        public void RecordTransition(float[] state, GridAction action, GridStepResult result, float[] nextState)
        {
            if (!recordingEnabled)
            {
                return;
            }

            var row = new GridTransitionRecord
            {
                episode = episodeIndex,
                step = result.steps,
                state = state,
                action = (int)action,
                reward = result.reward,
                next_state = nextState,
                done = result.done,
                eventName = result.eventName
            };

            File.AppendAllText(outputPath, JsonUtility.ToJson(row) + "\n", Encoding.UTF8);
        }

        public string OutputPath => outputPath;
    }
}
