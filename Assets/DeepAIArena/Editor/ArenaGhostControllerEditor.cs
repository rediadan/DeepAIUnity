using UnityEditor;
using UnityEngine;

namespace DeepAIArena.Editor
{
    [CustomEditor(typeof(ArenaGhostController))]
    public class ArenaGhostControllerEditor : UnityEditor.Editor
    {
        private SerializedProperty policyModeProperty;
        private SerializedProperty onnxPolicyProperty;

        private void OnEnable()
        {
            policyModeProperty = serializedObject.FindProperty("policyMode");
            onnxPolicyProperty = serializedObject.FindProperty("onnxPolicy");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.LabelField("Ghost Mode", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(policyModeProperty, new GUIContent("Mode"));

            var mode = (ArenaGhostPolicyMode)policyModeProperty.enumValueIndex;
            switch (mode)
            {
                case ArenaGhostPolicyMode.RuleBased:
                    EditorGUILayout.HelpBox(
                        "Rule-Based: handcrafted Ghost logic. Use this when collecting player logs for BC or DQN datasets.",
                        MessageType.Info);
                    break;

                case ArenaGhostPolicyMode.OnnxInference:
                    EditorGUILayout.HelpBox(
                        "ONNX Inference: runs the Behavior Cloning model. Expected outputs are move_logits and shove_prob.",
                        MessageType.Info);
                    DrawPolicyField("BC ONNX Policy");
                    break;

                case ArenaGhostPolicyMode.DqnInference:
                    EditorGUILayout.HelpBox(
                        "DQN Inference: runs the DQN model. Expected output is q_values, and the highest Q-value action is converted to Ghost input.",
                        MessageType.Info);
                    DrawPolicyField("DQN ONNX Policy");
                    break;
            }

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawPolicyField(string label)
        {
            EditorGUILayout.PropertyField(onnxPolicyProperty, new GUIContent(label));
            if (onnxPolicyProperty.objectReferenceValue == null)
            {
                EditorGUILayout.HelpBox(
                    "Assign an ArenaGhostOnnxPolicy component and set its ONNX model plus stats JSON.",
                    MessageType.Warning);
            }
        }
    }
}
