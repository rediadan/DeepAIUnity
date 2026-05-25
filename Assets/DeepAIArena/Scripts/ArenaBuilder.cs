using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

namespace DeepAIArena
{
    public static class ArenaBuilder
    {
        public static void Build(Transform parent)
        {
            EnsureCamera();

            var existingRoot = parent.Find("ArenaRoot");
            if (existingRoot != null)
            {
                if (Application.isPlaying)
                {
                    Object.Destroy(existingRoot.gameObject);
                }
                else
                {
                    Object.DestroyImmediate(existingRoot.gameObject);
                }
            }

            var root = new GameObject("ArenaRoot");
            root.transform.SetParent(parent, false);

            var managerObject = new GameObject("ArenaGameManager");
            managerObject.transform.SetParent(root.transform, false);
            var manager = managerObject.AddComponent<ArenaGameManager>();

            BuildSharedBaseArena(root.transform);

            var sharedBase = CreateSharedBase(root.transform, new Vector2(0f, -4.45f), new Color(0.95f, 0.86f, 0.3f));
            var item = CreateItem(root.transform, new Vector2(0f, 0f));
            var player = CreateActor(root.transform, "Player", new Vector2(-8.8f, 0f), ArenaSide.Left, new Color(0.31f, 0.87f, 1f), false);
            var ghost = CreateActor(root.transform, "Ghost", new Vector2(8.8f, 0f), ArenaSide.Right, new Color(1f, 0.4f, 0.45f), true);

            manager.Configure(player, ghost, item, sharedBase);
        }

        private static void EnsureCamera()
        {
            var camera = Object.FindAnyObjectByType<Camera>();
            if (camera == null)
            {
                var cameraObject = new GameObject("Main Camera");
                camera = cameraObject.AddComponent<Camera>();
                cameraObject.tag = "MainCamera";
            }

            camera.orthographic = true;
            camera.backgroundColor = new Color(0.08f, 0.1f, 0.14f);
            camera.orthographicSize = 6.1f;
            camera.transform.position = new Vector3(0f, 0f, -10f);
        }

        private static void BuildSharedBaseArena(Transform parent)
        {
            var wallColor = new Color(0.2f, 0.23f, 0.3f);
            var blockColor = new Color(0.34f, 0.39f, 0.5f);
            var laneColor = new Color(0.16f, 0.2f, 0.27f);

            CreateVisualZone(parent, new Vector2(0f, 3.25f), new Vector2(3.2f, 0.75f), new Color(0.18f, 0.25f, 0.33f), "TopItemZone");
            CreateVisualZone(parent, new Vector2(0f, 0f), new Vector2(3.0f, 0.9f), new Color(0.25f, 0.2f, 0.18f), "CenterItemZone");
            CreateVisualZone(parent, new Vector2(0f, -2.3f), new Vector2(3.2f, 0.75f), new Color(0.18f, 0.26f, 0.22f), "BottomItemZone");

            CreateObstacle(parent, new Vector2(0f, 5.35f), new Vector2(20.5f, 0.35f), wallColor, "TopWall");
            CreateObstacle(parent, new Vector2(0f, -5.35f), new Vector2(20.5f, 0.35f), wallColor, "BottomWall");
            CreateObstacle(parent, new Vector2(-10.15f, 0f), new Vector2(0.35f, 10.7f), wallColor, "LeftWall");
            CreateObstacle(parent, new Vector2(10.15f, 0f), new Vector2(0.35f, 10.7f), wallColor, "RightWall");

            CreateObstacle(parent, new Vector2(-4.8f, 2.25f), new Vector2(3.2f, 0.42f), blockColor, "TopLeftGate");
            CreateObstacle(parent, new Vector2(4.8f, 2.25f), new Vector2(3.2f, 0.42f), blockColor, "TopRightGate");
            CreateObstacle(parent, new Vector2(-4.8f, -1.25f), new Vector2(3.2f, 0.42f), blockColor, "BottomLeftGate");
            CreateObstacle(parent, new Vector2(4.8f, -1.25f), new Vector2(3.2f, 0.42f), blockColor, "BottomRightGate");

            CreateObstacle(parent, new Vector2(-1.6f, 1.15f), new Vector2(0.45f, 1.45f), laneColor, "CenterLeftPillar");
            CreateObstacle(parent, new Vector2(1.6f, 1.15f), new Vector2(0.45f, 1.45f), laneColor, "CenterRightPillar");
            CreateObstacle(parent, new Vector2(-1.6f, -1.15f), new Vector2(0.45f, 1.2f), laneColor, "LowerLeftPillar");
            CreateObstacle(parent, new Vector2(1.6f, -1.15f), new Vector2(0.45f, 1.2f), laneColor, "LowerRightPillar");

            CreateObstacle(parent, new Vector2(-7.15f, 0f), new Vector2(0.5f, 2.6f), blockColor, "LeftStartDivider");
            CreateObstacle(parent, new Vector2(7.15f, 0f), new Vector2(0.5f, 2.6f), blockColor, "RightStartDivider");

            BuildV2Complexity(parent);
        }

        public static void EnsureV2Environment(Transform arenaRoot)
        {
            if (arenaRoot == null || arenaRoot.Find("UpperShortcutDoor") != null)
            {
                return;
            }

            BuildV2Complexity(arenaRoot);
        }

        private static void BuildV2Complexity(Transform parent)
        {
            var upperDoor = CreateDoor(parent, new Vector2(0f, 1.15f), new Vector2(0.55f, 1.35f), "UpperShortcutDoor");
            var lowerDoor = CreateDoor(parent, new Vector2(0f, -1.15f), new Vector2(0.55f, 1.15f), "LowerShortcutDoor");
            CreateSwitch(parent, new Vector2(-3.2f, 0.75f), new Vector2(0.8f, 0.45f), upperDoor, "UpperDoorSwitch");
            CreateSwitch(parent, new Vector2(3.2f, -2.05f), new Vector2(0.8f, 0.45f), lowerDoor, "LowerDoorSwitch");

            CreateMovingObstacle(
                parent,
                new Vector2(-2.7f, 3.25f),
                new Vector2(2.7f, 3.25f),
                new Vector2(0.62f, 0.62f),
                new Color(0.85f, 0.25f, 0.68f),
                "TopMovingObstacle",
                2.0f);
            CreateMovingObstacle(
                parent,
                new Vector2(2.7f, -2.3f),
                new Vector2(-2.7f, -2.3f),
                new Vector2(0.62f, 0.62f),
                new Color(0.85f, 0.25f, 0.68f),
                "BottomMovingObstacle",
                1.75f);
        }

        private static void CreateObstacle(Transform parent, Vector2 position, Vector2 scale, Color color, string name)
        {
            var block = new GameObject(name);
            block.transform.SetParent(parent, false);
            block.transform.position = position;
            block.transform.localScale = scale;

            var renderer = block.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;

            block.AddComponent<BoxCollider2D>();
        }

        private static ArenaDoor CreateDoor(Transform parent, Vector2 position, Vector2 scale, string name)
        {
            var doorObject = new GameObject(name);
            doorObject.transform.SetParent(parent, false);
            doorObject.transform.position = position;
            doorObject.transform.localScale = scale;

            var renderer = doorObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = new Color(0.28f, 0.32f, 0.42f);
            renderer.sortingOrder = 4;

            doorObject.AddComponent<BoxCollider2D>();
            return doorObject.AddComponent<ArenaDoor>();
        }

        private static ArenaSwitch CreateSwitch(
            Transform parent,
            Vector2 position,
            Vector2 scale,
            ArenaDoor linkedDoor,
            string name)
        {
            var switchObject = new GameObject(name);
            switchObject.transform.SetParent(parent, false);
            switchObject.transform.position = position;
            switchObject.transform.localScale = scale;

            var renderer = switchObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = new Color(0.2f, 0.48f, 0.85f);
            renderer.sortingOrder = 3;

            var collider = switchObject.AddComponent<BoxCollider2D>();
            collider.isTrigger = true;

            var arenaSwitch = switchObject.AddComponent<ArenaSwitch>();
            arenaSwitch.SetLinkedDoor(linkedDoor);
            arenaSwitch.SetDoorOpenSeconds(linkedDoor != null ? linkedDoor.HoldOpenSeconds : 2.5f);
            return arenaSwitch;
        }

        private static ArenaMovingObstacle CreateMovingObstacle(
            Transform parent,
            Vector2 pointA,
            Vector2 pointB,
            Vector2 scale,
            Color color,
            string name,
            float speed)
        {
            var obstacleObject = new GameObject(name);
            obstacleObject.transform.SetParent(parent, false);
            obstacleObject.transform.position = pointA;
            obstacleObject.transform.localScale = scale;

            var renderer = obstacleObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;
            renderer.sortingOrder = 4;

            obstacleObject.AddComponent<BoxCollider2D>();
            var rigidbody = obstacleObject.AddComponent<Rigidbody2D>();
            rigidbody.gravityScale = 0f;
            rigidbody.freezeRotation = true;
            rigidbody.bodyType = RigidbodyType2D.Kinematic;

            var movingObstacle = obstacleObject.AddComponent<ArenaMovingObstacle>();
            movingObstacle.Configure(pointA, pointB, speed);
            movingObstacle.ResetMotion();
            return movingObstacle;
        }

        private static void CreateVisualZone(Transform parent, Vector2 position, Vector2 scale, Color color, string name)
        {
            var zone = new GameObject(name);
            zone.transform.SetParent(parent, false);
            zone.transform.position = position;
            zone.transform.localScale = scale;

            var renderer = zone.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;
            renderer.sortingOrder = -2;
        }

        private static ArenaBaseZone CreateSharedBase(Transform parent, Vector2 position, Color color)
        {
            var baseObject = new GameObject("SharedBase");
            baseObject.transform.SetParent(parent, false);
            baseObject.transform.position = position;
            baseObject.transform.localScale = new Vector3(3.5f, 1.05f, 1f);

            var renderer = baseObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;
            renderer.sortingOrder = 2;

            var collider = baseObject.AddComponent<BoxCollider2D>();
            collider.isTrigger = true;

            var zone = baseObject.AddComponent<ArenaBaseZone>();
            zone.SharedBase = true;
            zone.Side = ArenaSide.Left;
            return zone;
        }

        private static ArenaItem CreateItem(Transform parent, Vector2 position)
        {
            var itemObject = new GameObject("ArenaItem");
            itemObject.transform.SetParent(parent, false);
            itemObject.transform.position = position;
            itemObject.transform.localScale = new Vector3(0.7f, 0.7f, 1f);

            var renderer = itemObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = new Color(1f, 0.94f, 0.45f);
            renderer.sortingOrder = 5;

            var collider = itemObject.AddComponent<BoxCollider2D>();
            collider.isTrigger = true;

            return itemObject.AddComponent<ArenaItem>();
        }

        private static ArenaCharacterController CreateActor(Transform parent, string name, Vector2 position, ArenaSide side, Color color, bool isGhost)
        {
            var actorObject = new GameObject(name);
            actorObject.transform.SetParent(parent, false);
            actorObject.transform.position = position;
            actorObject.transform.localScale = new Vector3(0.42f, 0.42f, 1f);

            var renderer = actorObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;
            renderer.sortingOrder = 10;

            var collider = actorObject.AddComponent<CircleCollider2D>();
            collider.radius = 0.5f;
            var rigidbody = actorObject.AddComponent<Rigidbody2D>();
            rigidbody.gravityScale = 0f;
            rigidbody.freezeRotation = true;
            rigidbody.interpolation = RigidbodyInterpolation2D.Interpolate;

            var controller = actorObject.AddComponent<ArenaCharacterController>();
            controller.Side = side;
            controller.IsGhost = isGhost;

            actorObject.AddComponent<BehaviorParameters>();
            var decisionRequester = actorObject.AddComponent<DecisionRequester>();
            decisionRequester.enabled = false;
            var mlAgent = actorObject.AddComponent<ArenaMlAgent>();
            mlAgent.SetControlActive(false);

            var ghostController = actorObject.AddComponent<ArenaGhostController>();
            var onnxPolicy = actorObject.AddComponent<ArenaGhostOnnxPolicy>();
            ghostController.enabled = isGhost;
            onnxPolicy.enabled = isGhost;

            return controller;
        }
    }

}
