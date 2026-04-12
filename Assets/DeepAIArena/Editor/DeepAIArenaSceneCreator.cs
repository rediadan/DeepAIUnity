using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepAIArena.Editor
{
    public static class DeepAIArenaSceneCreator
    {
        [MenuItem("Tools/Deep AI/Create Arena Scene")]
        public static void CreateScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var bootstrapObject = new GameObject("DeepAIArenaBootstrap");
            var bootstrap = bootstrapObject.AddComponent<DeepAIArenaBootstrap>();
            var serializedBootstrap = new SerializedObject(bootstrap);
            serializedBootstrap.FindProperty("buildOnPlay").boolValue = false;
            serializedBootstrap.ApplyModifiedPropertiesWithoutUndo();

            ArenaBuilder.Build(bootstrapObject.transform);

            var sceneDirectory = "Assets/Scenes";
            if (!AssetDatabase.IsValidFolder(sceneDirectory))
            {
                Directory.CreateDirectory(Path.Combine(Application.dataPath, "Scenes"));
                AssetDatabase.Refresh();
            }

            var scenePath = $"{sceneDirectory}/DeepAIArena.unity";
            EditorSceneManager.SaveScene(scene, scenePath);

            AddSceneToBuildSettings(scenePath);
            Selection.activeGameObject = bootstrapObject;
            EditorGUIUtility.PingObject(bootstrapObject);
            Debug.Log($"Deep AI Arena scene created at {scenePath}");
        }

        private static void AddSceneToBuildSettings(string scenePath)
        {
            var currentScenes = EditorBuildSettings.scenes;
            foreach (var entry in currentScenes)
            {
                if (entry.path == scenePath)
                {
                    return;
                }
            }

            var updatedScenes = new EditorBuildSettingsScene[currentScenes.Length + 1];
            currentScenes.CopyTo(updatedScenes, 0);
            updatedScenes[^1] = new EditorBuildSettingsScene(scenePath, true);
            EditorBuildSettings.scenes = updatedScenes;
        }
    }
}
