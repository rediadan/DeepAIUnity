using System.Collections.Generic;
using UnityEngine;

namespace GridWorldExperiment
{
    public class StateEncoder : MonoBehaviour
    {
        public const int ChannelCount = 10;

        [SerializeField] private GridWorldManager manager;

        private void Reset()
        {
            manager = GetComponent<GridWorldManager>();
        }

        public float[] EncodeFlat()
        {
            if (manager == null)
            {
                manager = GetComponent<GridWorldManager>();
            }

            var size = manager.GridSize;
            var values = new float[ChannelCount * size * size];
            WritePosition(values, size, 0, manager.PlayerPosition, 1f);
            WritePositions(values, size, 1, manager.Coins, 1f);
            WritePositions(values, size, 2, manager.Enemies, 1f);
            WriteWalls(values, size, 3);
            WritePositions(values, size, 4, manager.Traps, 1f);
            WriteVisited(values, size, 5, manager.Visited, 1f);
            WriteDanger(values, size, 6, manager.Enemies, manager.DangerRadius);
            WritePositions(values, size, 7, manager.EnemyPrevious, 1f);
            WritePositions(values, size, 8, manager.EnemyPredicted, 1f);

            var targetCoin = manager.NearestCoin();
            if (targetCoin.HasValue)
            {
                WritePosition(values, size, 9, targetCoin.Value, 1f);
            }

            return values;
        }

        private void WritePositions(float[] values, int size, int channel, IReadOnlyList<Vector2> positions, float value)
        {
            foreach (var position in positions)
            {
                WritePosition(values, size, channel, position, value);
            }
        }

        private static void WriteVisited(float[] values, int size, int channel, IEnumerable<Vector2Int> cells, float value)
        {
            foreach (var cell in cells)
            {
                WriteCell(values, size, channel, cell.y, cell.x, value);
            }
        }

        private void WritePosition(float[] values, int size, int channel, Vector2 position, float value)
        {
            var cell = manager.WorldToGrid(position);
            WriteCell(values, size, channel, cell.y, cell.x, value);
        }

        private void WriteWalls(float[] values, int size, int channel)
        {
            for (var y = 0; y < size; y++)
            {
                for (var x = 0; x < size; x++)
                {
                    if (manager.IsWallCell(y, x))
                    {
                        WriteCell(values, size, channel, y, x, 1f);
                    }
                }
            }
        }

        private void WriteDanger(float[] values, int size, int channel, IReadOnlyList<Vector2> enemies, int radius)
        {
            var radiusWorld = Mathf.Max(1, radius) * manager.CellWorldSize;
            foreach (var enemy in enemies)
            {
                var center = manager.WorldToGrid(enemy);
                var search = Mathf.Max(1, radius);
                for (var y = Mathf.Max(0, center.y - search); y <= Mathf.Min(size - 1, center.y + search); y++)
                {
                    for (var x = Mathf.Max(0, center.x - search); x <= Mathf.Min(size - 1, center.x + search); x++)
                    {
                        var distance = Vector2.Distance(enemy, manager.CellCenterWorld(y, x));
                        if (distance > radiusWorld)
                        {
                            continue;
                        }

                        var index = channel * size * size + y * size + x;
                        values[index] = Mathf.Max(values[index], 1f - distance / Mathf.Max(radiusWorld, 0.0001f));
                    }
                }
            }
        }

        private static void WriteCell(float[] values, int size, int channel, int y, int x, float value)
        {
            if (x < 0 || y < 0 || x >= size || y >= size)
            {
                return;
            }

            var index = channel * size * size + y * size + x;
            values[index] = value;
        }
    }
}
