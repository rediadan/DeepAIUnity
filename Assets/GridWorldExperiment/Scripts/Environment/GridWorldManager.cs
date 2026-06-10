using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace GridWorldExperiment
{
    [RequireComponent(typeof(RewardCalculator))]
    [RequireComponent(typeof(StateEncoder))]
    public class GridWorldManager : MonoBehaviour
    {
        [Header("Continuous World")]
        [SerializeField] private float worldSize = 12f;
        [SerializeField] private int gridSize = 24;
        [SerializeField] private float moveStep = 0.25f;
        [SerializeField] private float playerRadius = 0.2f;
        [SerializeField] private float coinRadius = 0.22f;
        [SerializeField] private float enemyRadius = 0.24f;
        [SerializeField] private float trapRadius = 0.28f;
        [SerializeField] private float enemySpeed = 0.16f;

        [Header("Episode")]
        [SerializeField] private int maxSteps = 240;
        [SerializeField] private int targetScore = 5;
        [SerializeField] private int randomSeed = 42;
        [SerializeField] private int minCoins = 3;
        [SerializeField] private int maxCoins = 6;
        [SerializeField] private int minEnemies = 1;
        [SerializeField] private int maxEnemies = 3;
        [SerializeField] private float wallDensity = 0.12f;
        [SerializeField] private float trapDensity = 0.04f;
        [SerializeField] private int dangerRadius = 3;
        [SerializeField] private bool movingEnemies = true;

        [Header("Visualization")]
        [SerializeField] private bool drawWorldObjects = true;
        [SerializeField] private bool drawRasterOverlay;

        private readonly List<Vector2> coins = new();
        private readonly List<Vector2> enemies = new();
        private readonly List<Vector2> enemyPrevious = new();
        private readonly List<Vector2> enemyPredicted = new();
        private readonly List<Rect> walls = new();
        private readonly List<Vector2> traps = new();
        private readonly HashSet<Vector2Int> visited = new();
        private readonly List<GameObject> visuals = new();
        private RewardCalculator rewardCalculator;
        private System.Random random;
        private Vector2 playerPosition;
        private int score;
        private int steps;
        private bool done;
        private int currentSeed;

        public float WorldSize => worldSize;
        public int GridSize => gridSize;
        public int MaxSteps => maxSteps;
        public int DangerRadius => dangerRadius;
        public float PlayerRadius => playerRadius;
        public float CoinRadius => coinRadius;
        public float EnemyRadius => enemyRadius;
        public float TrapRadius => trapRadius;
        public float EnemySpeed => enemySpeed;
        public float CellWorldSize => worldSize / Mathf.Max(1, gridSize);
        public int Score => score;
        public int Steps => steps;
        public bool Done => done;
        public Vector2 PlayerPosition => playerPosition;
        public IReadOnlyList<Vector2> Coins => coins;
        public IReadOnlyList<Vector2> Enemies => enemies;
        public IReadOnlyList<Vector2> EnemyPrevious => enemyPrevious;
        public IReadOnlyList<Vector2> EnemyPredicted => enemyPredicted;
        public IReadOnlyList<Rect> Walls => walls;
        public IReadOnlyList<Vector2> Traps => traps;
        public IReadOnlyCollection<Vector2Int> Visited => visited;

        private void Awake()
        {
            rewardCalculator = GetComponent<RewardCalculator>();
            ResetEpisode(randomSeed);
        }

        public void ResetEpisode()
        {
            ResetEpisode(currentSeed + 1);
        }

        public void ResetEpisode(int seed)
        {
            currentSeed = seed;
            random = new System.Random(currentSeed);
            score = 0;
            steps = 0;
            done = false;
            coins.Clear();
            enemies.Clear();
            enemyPrevious.Clear();
            enemyPredicted.Clear();
            walls.Clear();
            traps.Clear();
            visited.Clear();
            playerPosition = Vector2.zero;

            GenerateWalls();
            for (var i = 0; i < TrapCount(); i++)
            {
                traps.Add(SampleOpenPosition(trapRadius));
            }

            playerPosition = SampleOpenPosition(playerRadius);
            visited.Add(WorldToGrid(playerPosition));

            var coinCount = random.Next(minCoins, maxCoins + 1);
            var enemyCount = random.Next(minEnemies, maxEnemies + 1);
            for (var i = 0; i < coinCount; i++)
            {
                coins.Add(SampleOpenPosition(coinRadius));
            }

            for (var i = 0; i < enemyCount; i++)
            {
                enemies.Add(SampleOpenPosition(enemyRadius));
            }

            enemyPrevious.AddRange(enemies);
            RefreshEnemyPredictions();
            RefreshVisuals();
        }

        public GridStepResult Step(GridAction action)
        {
            if (done)
            {
                return BuildResult(0f, true, "already_done");
            }

            steps++;
            var reward = rewardCalculator.StepReward;
            var eventName = "move";
            var hitWall = false;
            var hitTrap = false;
            var collectedCoin = false;
            var death = false;
            var next = playerPosition + ActionDelta(action) * moveStep;

            if (IsPositionBlocked(next, playerRadius))
            {
                hitWall = true;
                eventName = "wall_hit";
                reward += rewardCalculator.WallHitReward;
            }
            else
            {
                playerPosition = next;
                visited.Add(WorldToGrid(playerPosition));
            }

            if (TouchesAny(playerPosition, playerRadius, traps, trapRadius))
            {
                hitTrap = true;
                eventName = "trap";
                reward += rewardCalculator.TrapReward;
            }

            var coinIndex = FirstTouchingIndex(playerPosition, playerRadius, coins, coinRadius);
            if (coinIndex >= 0)
            {
                collectedCoin = true;
                eventName = "coin";
                coins.RemoveAt(coinIndex);
                score++;
                reward += rewardCalculator.CoinReward;
            }

            death = TouchesAny(playerPosition, playerRadius, enemies, enemyRadius);
            if (!death && movingEnemies)
            {
                MoveEnemies();
                death = TouchesAny(playerPosition, playerRadius, enemies, enemyRadius);
            }
            else
            {
                enemyPrevious.Clear();
                enemyPrevious.AddRange(enemies);
                RefreshEnemyPredictions();
            }

            if (death)
            {
                eventName = "death";
                reward += rewardCalculator.DeathReward;
                done = true;
            }
            else if (score >= targetScore)
            {
                eventName = "clear";
                reward += rewardCalculator.ClearBonusReward;
                done = true;
            }
            else if (steps >= maxSteps)
            {
                eventName = "timeout";
                done = true;
            }

            RefreshVisuals();
            var result = BuildResult(reward, done, eventName);
            result.hitWall = hitWall;
            result.hitTrap = hitTrap;
            result.collectedCoin = collectedCoin;
            result.death = death;
            return result;
        }

        public Vector2Int WorldToGrid(Vector2 position)
        {
            var scale = gridSize / Mathf.Max(worldSize, 0.0001f);
            var x = Mathf.Clamp(Mathf.FloorToInt(position.x * scale), 0, gridSize - 1);
            var y = Mathf.Clamp(Mathf.FloorToInt(position.y * scale), 0, gridSize - 1);
            return new Vector2Int(x, y);
        }

        public Vector2 CellCenterWorld(int y, int x)
        {
            var cell = CellWorldSize;
            return new Vector2((x + 0.5f) * cell, (y + 0.5f) * cell);
        }

        public Vector3 WorldToScene(Vector2 position)
        {
            var offset = worldSize * 0.5f;
            return transform.position + new Vector3(position.x - offset, position.y - offset, 0f);
        }

        public Vector3 CellToWorld(Vector2Int cell)
        {
            return WorldToScene(CellCenterWorld(cell.y, cell.x));
        }

        public bool IsPositionBlocked(Vector2 position, float radius)
        {
            if (position.x < radius || position.y < radius || position.x > worldSize - radius || position.y > worldSize - radius)
            {
                return true;
            }

            return walls.Any(wall => CircleIntersectsRect(position, radius, wall));
        }

        public Vector2? NearestCoin()
        {
            if (coins.Count == 0)
            {
                return null;
            }

            return coins.OrderBy(coin => Vector2.Distance(playerPosition, coin)).First();
        }

        public bool IsWallCell(int y, int x)
        {
            var center = CellCenterWorld(y, x);
            return walls.Any(wall => wall.Contains(center));
        }

        private GridStepResult BuildResult(float reward, bool isDone, string eventName)
        {
            return new GridStepResult
            {
                reward = reward,
                done = isDone,
                eventName = eventName,
                score = score,
                steps = steps
            };
        }

        private void GenerateWalls()
        {
            var targetCount = WallCount();
            var attempts = 0;
            while (walls.Count < targetCount && attempts < targetCount * 80)
            {
                attempts++;
                var longAxis = random.NextDouble() < 0.5;
                var width = longAxis ? Range(1.1f, 2.4f) : Range(0.35f, 0.75f);
                var height = longAxis ? Range(0.35f, 0.75f) : Range(1.1f, 2.4f);
                var margin = Mathf.Max(width, height) * 0.5f + 0.4f;
                var center = new Vector2(Range(margin, worldSize - margin), Range(margin, worldSize - margin));
                var rect = Rect.MinMaxRect(center.x - width * 0.5f, center.y - height * 0.5f, center.x + width * 0.5f, center.y + height * 0.5f);
                if (Vector2.Distance(center, Vector2.one * worldSize * 0.5f) >= 0.8f)
                {
                    walls.Add(rect);
                }
            }
        }

        private int WallCount()
        {
            return Mathf.Max(3, Mathf.RoundToInt(gridSize * gridSize * wallDensity / 16f));
        }

        private int TrapCount()
        {
            return Mathf.Max(2, Mathf.RoundToInt(gridSize * gridSize * trapDensity / 6f));
        }

        private Vector2 SampleOpenPosition(float radius)
        {
            for (var i = 0; i < 2000; i++)
            {
                var candidate = new Vector2(Range(radius, worldSize - radius), Range(radius, worldSize - radius));
                if (!IsOccupied(candidate, radius))
                {
                    return candidate;
                }
            }

            throw new InvalidOperationException("Not enough open space to generate continuous GridWorld episode.");
        }

        private bool IsOccupied(Vector2 position, float radius)
        {
            return IsPositionBlocked(position, radius)
                || TouchesAny(position, radius, traps, trapRadius + 0.2f)
                || TouchesAny(position, radius, coins, coinRadius + 0.25f)
                || TouchesAny(position, radius, enemies, enemyRadius + 0.35f)
                || (playerPosition != Vector2.zero && Vector2.Distance(position, playerPosition) < radius + playerRadius + 0.35f);
        }

        private void MoveEnemies()
        {
            enemyPrevious.Clear();
            enemyPrevious.AddRange(enemies);
            for (var i = 0; i < enemies.Count; i++)
            {
                var enemy = enemies[i];
                var direction = playerPosition - enemy;
                if (direction.sqrMagnitude <= 0.000001f)
                {
                    continue;
                }

                var candidate = enemy + direction.normalized * enemySpeed;
                if (!IsPositionBlocked(candidate, enemyRadius))
                {
                    enemies[i] = candidate;
                }
            }

            RefreshEnemyPredictions();
        }

        private void RefreshEnemyPredictions()
        {
            enemyPredicted.Clear();
            foreach (var enemy in enemies)
            {
                var direction = playerPosition - enemy;
                if (direction.sqrMagnitude <= 0.000001f)
                {
                    enemyPredicted.Add(enemy);
                    continue;
                }

                var candidate = enemy + direction.normalized * enemySpeed;
                enemyPredicted.Add(IsPositionBlocked(candidate, enemyRadius) ? enemy : candidate);
            }
        }

        private float Range(float min, float max)
        {
            return (float)(min + random.NextDouble() * (max - min));
        }

        private static Vector2 ActionDelta(GridAction action)
        {
            return action switch
            {
                GridAction.Up => Vector2.up,
                GridAction.Down => Vector2.down,
                GridAction.Left => Vector2.left,
                GridAction.Right => Vector2.right,
                _ => Vector2.zero
            };
        }

        private static bool CircleIntersectsRect(Vector2 position, float radius, Rect rect)
        {
            var closestX = Mathf.Clamp(position.x, rect.xMin, rect.xMax);
            var closestY = Mathf.Clamp(position.y, rect.yMin, rect.yMax);
            var dx = position.x - closestX;
            var dy = position.y - closestY;
            return dx * dx + dy * dy <= radius * radius;
        }

        private static bool TouchesAny(Vector2 position, float radius, IReadOnlyList<Vector2> targets, float targetRadius)
        {
            for (var i = 0; i < targets.Count; i++)
            {
                if (Vector2.Distance(position, targets[i]) <= radius + targetRadius)
                {
                    return true;
                }
            }

            return false;
        }

        private static int FirstTouchingIndex(Vector2 position, float radius, IReadOnlyList<Vector2> targets, float targetRadius)
        {
            for (var i = 0; i < targets.Count; i++)
            {
                if (Vector2.Distance(position, targets[i]) <= radius + targetRadius)
                {
                    return i;
                }
            }

            return -1;
        }

        private void RefreshVisuals()
        {
            foreach (var visual in visuals)
            {
                if (visual != null)
                {
                    Destroy(visual);
                }
            }
            visuals.Clear();

            if (drawRasterOverlay)
            {
                DrawRasterOverlay();
            }

            if (!drawWorldObjects)
            {
                return;
            }

            foreach (var wall in walls)
            {
                visuals.Add(CreateRectVisual("Wall", wall, new Color(0.35f, 0.38f, 0.43f, 1f)));
            }

            foreach (var trap in traps)
            {
                visuals.Add(CreateCircleVisual("Trap", trap, trapRadius, new Color(0.85f, 0.27f, 0.2f, 1f)));
            }

            foreach (var coin in coins)
            {
                visuals.Add(CreateCircleVisual("Coin", coin, coinRadius, new Color(0.95f, 0.78f, 0.18f, 1f)));
            }

            foreach (var enemy in enemies)
            {
                visuals.Add(CreateCircleVisual("Enemy", enemy, enemyRadius, new Color(0.9f, 0.12f, 0.18f, 1f)));
            }

            foreach (var predicted in enemyPredicted)
            {
                visuals.Add(CreateCircleVisual("Enemy Predicted", predicted, enemyRadius * 0.45f, new Color(1f, 0.35f, 0.2f, 0.65f)));
            }

            visuals.Add(CreateCircleVisual("Player", playerPosition, playerRadius, new Color(0.25f, 0.65f, 1f, 1f)));
        }

        private void DrawRasterOverlay()
        {
            var cellSize = CellWorldSize;
            foreach (var cell in visited)
            {
                var rect = Rect.MinMaxRect(
                    cell.x * cellSize,
                    cell.y * cellSize,
                    (cell.x + 1) * cellSize,
                    (cell.y + 1) * cellSize);
                visuals.Add(CreateRectVisual("Visited Cell", rect, new Color(0.18f, 0.2f, 0.24f, 0.55f)));
            }
        }

        private GameObject CreateCircleVisual(string objectName, Vector2 position, float radius, Color color)
        {
            var visual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            visual.name = objectName;
            visual.transform.SetParent(transform, false);
            visual.transform.position = WorldToScene(position);
            visual.transform.localScale = new Vector3(radius * 2f, radius * 2f, 0.08f);
            ApplyMaterialAndRemoveCollider(visual, color);
            return visual;
        }

        private GameObject CreateRectVisual(string objectName, Rect rect, Color color)
        {
            var visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visual.name = objectName;
            visual.transform.SetParent(transform, false);
            visual.transform.position = WorldToScene(rect.center);
            visual.transform.localScale = new Vector3(rect.width, rect.height, 0.08f);
            ApplyMaterialAndRemoveCollider(visual, color);
            return visual;
        }

        private static void ApplyMaterialAndRemoveCollider(GameObject visual, Color color)
        {
            var renderer = visual.GetComponent<Renderer>();
            renderer.sharedMaterial = new Material(Shader.Find("Sprites/Default"))
            {
                color = color
            };
            var collider = visual.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(12, 12, 420, 120), GUI.skin.box);
            GUILayout.Label($"Continuous Raster World seed={currentSeed} score={score} steps={steps}/{maxSteps}");
            GUILayout.Label($"grid={gridSize}x{gridSize} channels={StateEncoder.ChannelCount} coins={coins.Count} enemies={enemies.Count} done={done}");
            GUILayout.EndArea();
        }
    }
}
