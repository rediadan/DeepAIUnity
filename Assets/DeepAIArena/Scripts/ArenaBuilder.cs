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

            BuildSharedBaseArena(root.transform);

            var sharedBase = CreateSharedBase(root.transform, new Vector2(0f, -4.05f), new Color(0.95f, 0.86f, 0.3f));
            var item = CreateItem(root.transform, new Vector2(0f, 4.55f));
            var player = CreateActor(root.transform, "Player", new Vector2(-9.75f, 5.15f), ArenaSide.Left, new Color(0.31f, 0.87f, 1f), false);
            var ghost = CreateActor(root.transform, "Ghost", new Vector2(9.75f, 5.15f), ArenaSide.Right, new Color(1f, 0.4f, 0.45f), true);

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
            camera.orthographicSize = 7f;
            camera.backgroundColor = new Color(0.08f, 0.1f, 0.14f);
            camera.orthographicSize = 7.2f;
            camera.transform.position = new Vector3(0f, 0.4f, -10f);
        }

        private static void BuildSharedBaseArena(Transform parent)
        {
            CreateFloor(parent, new Vector2(0f, -5.05f), new Vector2(31f, 0.75f), new Color(0.18f, 0.2f, 0.26f), PlatformTraversalMode.Neither);
            CreateFloor(parent, new Vector2(-10.15f, 4.55f), new Vector2(3.2f, 0.26f), new Color(0.27f, 0.73f, 0.93f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(10.15f, 4.55f), new Vector2(3.2f, 0.26f), new Color(0.95f, 0.52f, 0.58f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(0f, 4.35f), new Vector2(2.3f, 0.22f), new Color(0.92f, 0.83f, 0.32f), PlatformTraversalMode.Both);

            CreateFloor(parent, new Vector2(-6.6f, 3.25f), new Vector2(2.4f, 0.22f), new Color(0.45f, 0.52f, 0.92f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(-2.8f, 2.1f), new Vector2(2.4f, 0.22f), new Color(0.45f, 0.52f, 0.92f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(2.8f, 2.1f), new Vector2(2.4f, 0.22f), new Color(0.45f, 0.52f, 0.92f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(6.6f, 3.25f), new Vector2(2.4f, 0.22f), new Color(0.45f, 0.52f, 0.92f), PlatformTraversalMode.Both);

            CreateFloor(parent, new Vector2(0f, 1.25f), new Vector2(3.2f, 0.24f), new Color(0.78f, 0.41f, 0.28f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(-7.2f, 0.1f), new Vector2(3.1f, 0.22f), new Color(0.22f, 0.68f, 0.55f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(7.2f, 0.1f), new Vector2(3.1f, 0.22f), new Color(0.22f, 0.68f, 0.55f), PlatformTraversalMode.Both);

            CreateFloor(parent, new Vector2(-5.1f, -2.15f), new Vector2(4.1f, 0.24f), new Color(0.19f, 0.57f, 0.44f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(5.1f, -2.15f), new Vector2(4.1f, 0.24f), new Color(0.19f, 0.57f, 0.44f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(0f, -2.85f), new Vector2(2.2f, 0.22f), new Color(0.84f, 0.48f, 0.26f), PlatformTraversalMode.Both);

            CreateFloor(parent, new Vector2(-1.7f, -4.35f), new Vector2(1.45f, 0.2f), new Color(0.85f, 0.77f, 0.29f), PlatformTraversalMode.Both);
            CreateFloor(parent, new Vector2(1.7f, -4.35f), new Vector2(1.45f, 0.2f), new Color(0.85f, 0.77f, 0.29f), PlatformTraversalMode.Both);
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
                actorObject.AddComponent<ArenaGhostOnnxPolicy>();
            }

            return controller;
        }
    }

}
