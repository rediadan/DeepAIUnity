using UnityEngine;

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

            CreateFloor(root.transform, new Vector2(0f, -4.5f), new Vector2(32f, 0.65f), new Color(0.2f, 0.22f, 0.28f), PlatformTraversalMode.Neither);
            CreateFloor(root.transform, new Vector2(0f, 0f), new Vector2(20f, 0.45f), new Color(0.32f, 0.46f, 0.78f), PlatformTraversalMode.Both);
            CreateFloor(root.transform, new Vector2(0f, -2.75f), new Vector2(24f, 0.4f), new Color(0.23f, 0.58f, 0.45f), PlatformTraversalMode.Both);

            CreateSteps(root.transform, -10.5f, true);
            CreateSteps(root.transform, 10.5f, false);
            CreateLowSteps(root.transform, -10.5f, true);
            CreateLowSteps(root.transform, 10.5f, false);

            CreateFloor(root.transform, new Vector2(-5f, 3f), new Vector2(5f, 0.35f), new Color(0.79f, 0.63f, 0.25f), PlatformTraversalMode.Both);
            CreateFloor(root.transform, new Vector2(5f, 3f), new Vector2(5f, 0.35f), new Color(0.79f, 0.63f, 0.25f), PlatformTraversalMode.Both);

            CreateFloor(root.transform, new Vector2(0f, 1.15f), new Vector2(3f, 0.3f), new Color(0.72f, 0.36f, 0.24f), PlatformTraversalMode.Both);
            CreateFloor(root.transform, new Vector2(0f, 2.25f), new Vector2(1.2f, 0.25f), new Color(0.95f, 0.82f, 0.36f), PlatformTraversalMode.Both);

            var leftBase = CreateBase(root.transform, "PlayerBase", new Vector2(-13f, 0.75f), ArenaSide.Left, new Color(0.28f, 0.78f, 0.96f));
            var rightBase = CreateBase(root.transform, "GhostBase", new Vector2(13f, 0.75f), ArenaSide.Right, new Color(0.95f, 0.37f, 0.41f));

            var item = CreateItem(root.transform);
            CreateFloor(root.transform, new Vector2(-12f, 0.7f), new Vector2(2.2f, 0.22f), new Color(0.42f, 0.72f, 0.95f), PlatformTraversalMode.Both);
            CreateFloor(root.transform, new Vector2(12f, 0.7f), new Vector2(2.2f, 0.22f), new Color(0.95f, 0.56f, 0.61f), PlatformTraversalMode.Both);

            var player = CreateActor(root.transform, "Player", new Vector2(-12f, 1.25f), ArenaSide.Left, new Color(0.31f, 0.87f, 1f), false);
            var ghost = CreateActor(root.transform, "Ghost", new Vector2(12f, 1.25f), ArenaSide.Right, new Color(1f, 0.4f, 0.45f), true);

            manager.Configure(player, ghost, item, leftBase, rightBase);
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
            camera.orthographicSize = 7f;
            camera.backgroundColor = new Color(0.08f, 0.1f, 0.14f);
            camera.transform.position = new Vector3(0f, 0f, -10f);
        }

        private static void CreateSteps(Transform parent, float startX, bool left)
        {
            var direction = left ? 1f : -1f;
            CreateFloor(parent, new Vector2(startX + direction * 1.2f, 1f), new Vector2(1.6f, 0.25f), new Color(0.58f, 0.4f, 0.86f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(startX + direction * 4.6f, 2.1f), new Vector2(1.35f, 0.25f), new Color(0.58f, 0.4f, 0.86f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(startX + direction * 8.2f, 3f), new Vector2(1.25f, 0.25f), new Color(0.58f, 0.4f, 0.86f), PlatformTraversalMode.Both);
        }

        private static void CreateLowSteps(Transform parent, float startX, bool left)
        {
            var direction = left ? 1f : -1f;
            CreateFloor(parent, new Vector2(startX + direction * 1.4f, -0.8f), new Vector2(1.7f, 0.22f), new Color(0.18f, 0.67f, 0.55f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(startX + direction * 4.8f, -1.8f), new Vector2(1.4f, 0.22f), new Color(0.18f, 0.67f, 0.55f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(startX + direction * 8.4f, -2.75f), new Vector2(1.25f, 0.22f), new Color(0.18f, 0.67f, 0.55f), PlatformTraversalMode.Both);
        }

        private static void CreateFloor(Transform parent, Vector2 position, Vector2 scale, Color color, PlatformTraversalMode traversalMode)
        {
            var block = new GameObject($"Block_{position.x}_{position.y}");
            block.transform.SetParent(parent, false);
            block.transform.position = position;
            block.transform.localScale = scale;

            var renderer = block.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;

            var collider = block.AddComponent<BoxCollider2D>();
            var platform = block.AddComponent<ArenaPlatform>();
            platform.SetTraversalMode(traversalMode);
            platform.ApplySettings(collider);
        }

        private static ArenaBaseZone CreateBase(Transform parent, string name, Vector2 position, ArenaSide side, Color color)
        {
            var baseObject = new GameObject(name);
            baseObject.transform.SetParent(parent, false);
            baseObject.transform.position = position;
            baseObject.transform.localScale = new Vector3(2f, 3f, 1f);

            var renderer = baseObject.AddComponent<SpriteRenderer>();
            renderer.sprite = RuntimeSpriteLibrary.Square;
            renderer.color = color;

            var collider = baseObject.AddComponent<BoxCollider2D>();
            collider.isTrigger = true;

            var zone = baseObject.AddComponent<ArenaBaseZone>();
            zone.Side = side;
            return zone;
        }

        private static ArenaItem CreateItem(Transform parent)
        {
            var itemObject = new GameObject("ArenaItem");
            itemObject.transform.SetParent(parent, false);
            itemObject.transform.position = new Vector3(0f, 2.9f, 0f);
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
            rigidbody.freezeRotation = true;
            rigidbody.interpolation = RigidbodyInterpolation2D.Interpolate;

            var controller = actorObject.AddComponent<ArenaCharacterController>();
            controller.Side = side;
            controller.IsGhost = isGhost;

            var sensor = new GameObject("GroundCheck");
            sensor.transform.SetParent(actorObject.transform, false);
            sensor.transform.localPosition = new Vector3(0f, -0.34f, 0f);
            controller.GroundCheck = sensor.transform;

            if (isGhost)
            {
                actorObject.AddComponent<ArenaGhostController>();
            }

            return controller;
        }
    }

}
