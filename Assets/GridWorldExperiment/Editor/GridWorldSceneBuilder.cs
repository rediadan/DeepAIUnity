#if UNITY_EDITOR
using GridWorldExperiment;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace GridWorldExperimentEditor
{
    public static class GridWorldSceneBuilder
    {
        [MenuItem("Tools/GridWorld/Create Demo Scene")]
        public static void CreateDemoScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            scene.name = "GridWorldDemo";

            var root = new GameObject("GridWorld");
            root.AddComponent<RewardCalculator>();
            root.AddComponent<StateEncoder>();
            root.AddComponent<GridWorldManager>();
            root.AddComponent<DemonstrationRecorder>();
            root.AddComponent<HumanController>();

            var camera = Camera.main;
            if (camera != null)
            {
                camera.transform.position = new Vector3(0f, 0f, -10f);
                camera.orthographic = true;
                camera.orthographicSize = 7.2f;
            }

            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(scene);
            Debug.Log("Created continuous GridWorld demo scene. Add SentisDqnAgent and assign a 10x24x24 ONNX model for inference mode.");
        }
    }
}
#endif
