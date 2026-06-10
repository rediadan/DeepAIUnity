using UnityEngine;
using UnityEngine.InputSystem;

namespace GridWorldExperiment
{
    public class HumanController : MonoBehaviour
    {
        [SerializeField] private GridWorldManager manager;
        [SerializeField] private StateEncoder stateEncoder;
        [SerializeField] private DemonstrationRecorder recorder;
        [SerializeField] private float keyRepeatSeconds = 0.12f;
        [SerializeField] private bool resetOnEpisodeDone = true;

        private float nextInputTime;

        private void Reset()
        {
            manager = GetComponent<GridWorldManager>();
            stateEncoder = GetComponent<StateEncoder>();
            recorder = GetComponent<DemonstrationRecorder>();
        }

        private void Awake()
        {
            if (manager == null) manager = GetComponent<GridWorldManager>();
            if (stateEncoder == null) stateEncoder = GetComponent<StateEncoder>();
            if (recorder == null) recorder = GetComponent<DemonstrationRecorder>();
        }

        private void Update()
        {
            if (Time.time < nextInputTime)
            {
                return;
            }

            if (Keyboard.current == null)
            {
                return;
            }

            if (manager.Done)
            {
                if (resetOnEpisodeDone && Keyboard.current.spaceKey.wasPressedThisFrame)
                {
                    recorder?.BeginNewEpisode();
                    manager.ResetEpisode();
                }
                return;
            }

            if (!TryReadAction(out var action))
            {
                return;
            }

            var state = stateEncoder.EncodeFlat();
            var result = manager.Step(action);
            var nextState = stateEncoder.EncodeFlat();
            recorder?.RecordTransition(state, action, result, nextState);
            nextInputTime = Time.time + keyRepeatSeconds;
        }

        private static bool TryReadAction(out GridAction action)
        {
            var keyboard = Keyboard.current;
            if (keyboard.upArrowKey.isPressed || keyboard.wKey.isPressed)
            {
                action = GridAction.Up;
                return true;
            }

            if (keyboard.downArrowKey.isPressed || keyboard.sKey.isPressed)
            {
                action = GridAction.Down;
                return true;
            }

            if (keyboard.leftArrowKey.isPressed || keyboard.aKey.isPressed)
            {
                action = GridAction.Left;
                return true;
            }

            if (keyboard.rightArrowKey.isPressed || keyboard.dKey.isPressed)
            {
                action = GridAction.Right;
                return true;
            }

            action = GridAction.Up;
            return false;
        }
    }
}
