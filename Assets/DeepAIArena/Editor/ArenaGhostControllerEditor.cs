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
                        "Rule-Based: 플레이어 로그를 수집할 때 쓰는 기본 Ghost입니다. 수작업 규칙으로 이동, 점프, 드롭, 밀치기를 결정합니다.",
                        MessageType.Info);
                    break;

                case ArenaGhostPolicyMode.OnnxInference:
                    EditorGUILayout.HelpBox(
                        "ONNX Input: 학습된 ONNX 모델로 Ghost를 움직입니다. 테스트나 비교 평가에 사용하면 됩니다.",
                        MessageType.Info);
                    EditorGUILayout.PropertyField(onnxPolicyProperty, new GUIContent("ONNX Policy"));
                    if (onnxPolicyProperty.objectReferenceValue == null)
                    {
                        EditorGUILayout.HelpBox(
                            "같은 오브젝트의 ArenaGhostOnnxPolicy를 연결하고, 그 안에 ONNX와 stats JSON을 넣어야 합니다.",
                            MessageType.Warning);
                    }
                    break;
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
