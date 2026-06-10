using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace DeepAIArena
{
    public enum ArenaRasterInputMode
    {
        SemanticChannels,
        SemanticRgbScreen,
        SemanticRgbFrameStack4
    }

    public class ArenaRasterEncoder : MonoBehaviour
    {
        public const int ChannelCount = 12;
        public const int RgbChannelCount = 3;
        public const int FrameStackSize = 4;
        public const int FrameStackChannelCount = RgbChannelCount * FrameStackSize;
        public const int RasterHeight = 32;
        public const int RasterWidth = 48;

        [SerializeField] private ArenaGameManager manager;
        [SerializeField] private Transform arenaRoot;
        [SerializeField] private ArenaRasterInputMode inputMode = ArenaRasterInputMode.SemanticRgbFrameStack4;
        [SerializeField] private Vector2 worldMin = new Vector2(-10.4f, -5.6f);
        [SerializeField] private Vector2 worldMax = new Vector2(10.4f, 5.6f);
        [SerializeField] private float actorDangerRadius = 0.9f;
        [SerializeField] private float movingObstacleDangerRadius = 1.15f;
        [Header("Target Highlight")]
        [SerializeField] private bool drawTargetHighlight = true;
        [SerializeField, Range(1, 4)] private int targetHighlightRadiusCells = 2;
        [SerializeField] private bool drawDebugOverlay;
        [Header("Debug PNG Dump")]
        [SerializeField] private bool dumpDebugPngOnInterval;
        [SerializeField] private ArenaSide debugDumpPerspective = ArenaSide.Left;
        [SerializeField, Min(0.05f)] private float debugDumpIntervalSeconds = 0.5f;
        [SerializeField, Range(1, 16)] private int debugDumpScale = 8;

        private readonly List<BoxCollider2D> staticObstacleCache = new();
        private readonly List<GameObject> overlayCells = new();
        private readonly List<float[]> leftRgbFrameStack = new(FrameStackSize);
        private readonly List<float[]> rightRgbFrameStack = new(FrameStackSize);
        private float nextDebugDumpTime;
        private int debugDumpFrameIndex;

        public ArenaRasterInputMode InputMode => inputMode;
        public int InputChannelCount => inputMode switch
        {
            ArenaRasterInputMode.SemanticRgbScreen => RgbChannelCount,
            ArenaRasterInputMode.SemanticRgbFrameStack4 => FrameStackChannelCount,
            _ => ChannelCount
        };
        public int InputLength => InputChannelCount * RasterHeight * RasterWidth;

        private void Reset()
        {
            manager = GetComponent<ArenaGameManager>();
            arenaRoot = transform.parent;
        }

        private void Awake()
        {
            if (manager == null)
            {
                manager = GetComponent<ArenaGameManager>();
            }

            if (arenaRoot == null && manager != null)
            {
                arenaRoot = manager.transform.parent;
            }

            RefreshStaticObstacleCache();
        }

        private void Update()
        {
            if (!dumpDebugPngOnInterval || !Application.isPlaying)
            {
                return;
            }

            if (Time.unscaledTime < nextDebugDumpTime)
            {
                return;
            }

            nextDebugDumpTime = Time.unscaledTime + Mathf.Max(0.05f, debugDumpIntervalSeconds);
            DumpRasterPreviewPng(debugDumpPerspective);
        }

        public float[] EncodeFlat(ArenaSide perspective)
        {
            var semanticChannels = EncodeSemanticChannelsFlat(perspective, out var observation);
            return inputMode switch
            {
                ArenaRasterInputMode.SemanticRgbScreen => ConvertSemanticChannelsToRgb(semanticChannels, observation),
                ArenaRasterInputMode.SemanticRgbFrameStack4 => EncodeSemanticRgbFrameStack(perspective, semanticChannels, observation),
                _ => semanticChannels
            };
        }

        public void ResetFrameStack()
        {
            leftRgbFrameStack.Clear();
            rightRgbFrameStack.Clear();
        }

        public void ResetFrameStack(ArenaSide perspective)
        {
            GetRgbFrameStack(perspective).Clear();
        }

        public float[] EncodeSemanticChannelsFlat(ArenaSide perspective)
        {
            return EncodeSemanticChannelsFlat(perspective, out _);
        }

        private float[] EncodeSemanticChannelsFlat(ArenaSide perspective, out ArenaObservationSnapshot observation)
        {
            if (manager == null)
            {
                manager = GetComponent<ArenaGameManager>();
            }

            if (staticObstacleCache.Count == 0)
            {
                RefreshStaticObstacleCache();
            }

            var values = new float[ChannelCount * RasterHeight * RasterWidth];
            observation = manager.BuildObservation(perspective);

            WritePoint(values, 0, observation.selfPosition, 1f);
            WritePoint(values, 1, observation.opponentPosition, 1f);
            WritePoint(values, 2, observation.itemPosition, 1f);
            WritePoint(values, 3, observation.basePosition, 1f);
            WriteStaticObstacles(values, 4);
            WriteWalkable(values, 5);
            WriteDoors(values, 6);
            WriteSwitches(values, 7);
            WriteMovingObstacles(values, 8);
            WriteDanger(values, 9, observation.opponentPosition, actorDangerRadius);
            WriteMovingObstacleDanger(values, 9);

            if (observation.selfHasItem)
            {
                FillChannel(values, 10, 1f);
            }

            if (observation.opponentHasItem)
            {
                FillChannel(values, 11, 1f);
            }

            return values;
        }

        public float[] EncodeSemanticRgbFlat(ArenaSide perspective)
        {
            var semanticChannels = EncodeSemanticChannelsFlat(perspective, out var observation);
            return ConvertSemanticChannelsToRgb(semanticChannels, observation);
        }

        [ContextMenu("Dump Raster Preview PNG/Left")]
        private void DumpLeftRasterPreviewPng()
        {
            DumpRasterPreviewPng(ArenaSide.Left);
        }

        [ContextMenu("Dump Raster Preview PNG/Right")]
        private void DumpRightRasterPreviewPng()
        {
            DumpRasterPreviewPng(ArenaSide.Right);
        }

        public string DumpRasterPreviewPng(ArenaSide perspective)
        {
            if (!TryGetCurrentPreviewInput(perspective, out var values, out var channelCount))
            {
                return string.Empty;
            }

            var outputDirectory = Path.Combine(Application.persistentDataPath, "raster_debug");
            Directory.CreateDirectory(outputDirectory);

            var safeMode = inputMode.ToString();
            var fileName = $"{safeMode}_{perspective}_{debugDumpFrameIndex:000000}.png";
            debugDumpFrameIndex++;
            var path = Path.Combine(outputDirectory, fileName);
            var texture = CreatePreviewTexture(values, channelCount, Mathf.Max(1, debugDumpScale));
            File.WriteAllBytes(path, texture.EncodeToPNG());
            if (Application.isPlaying)
            {
                Destroy(texture);
            }
            else
            {
                DestroyImmediate(texture);
            }

            Debug.Log($"[ArenaRasterEncoder] Saved raster preview: {path}", this);
            return path;
        }

        private bool TryGetCurrentPreviewInput(ArenaSide perspective, out float[] values, out int channelCount)
        {
            channelCount = InputChannelCount;
            if (inputMode == ArenaRasterInputMode.SemanticChannels)
            {
                values = EncodeSemanticRgbFlat(perspective);
                channelCount = RgbChannelCount;
                return values != null && values.Length == channelCount * RasterHeight * RasterWidth;
            }

            if (inputMode == ArenaRasterInputMode.SemanticRgbFrameStack4)
            {
                var stack = GetRgbFrameStack(perspective);
                if (stack.Count > 0)
                {
                    values = new float[FrameStackChannelCount * RasterHeight * RasterWidth];
                    var frameLength = RgbChannelCount * RasterHeight * RasterWidth;
                    for (var i = 0; i < FrameStackSize; i++)
                    {
                        var frame = stack[Mathf.Clamp(i - (FrameStackSize - stack.Count), 0, stack.Count - 1)];
                        Array.Copy(frame, 0, values, i * frameLength, frameLength);
                    }

                    return true;
                }
            }

            values = EncodeFlat(perspective);
            return values != null && values.Length == channelCount * RasterHeight * RasterWidth;
        }

        private static Texture2D CreatePreviewTexture(float[] values, int channelCount, int scale)
        {
            var frameCount = Mathf.Max(1, channelCount / RgbChannelCount);
            var outputWidth = RasterWidth * scale * frameCount;
            var outputHeight = RasterHeight * scale;
            var texture = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);
            var frameLength = RgbChannelCount * RasterHeight * RasterWidth;

            for (var frameIndex = 0; frameIndex < frameCount; frameIndex++)
            {
                var frameOffset = frameIndex * frameLength;
                for (var y = 0; y < RasterHeight; y++)
                {
                    for (var x = 0; x < RasterWidth; x++)
                    {
                        var cell = y * RasterWidth + x;
                        var color = new Color(
                            Mathf.Clamp01(values[frameOffset + cell]),
                            Mathf.Clamp01(values[frameOffset + RasterHeight * RasterWidth + cell]),
                            Mathf.Clamp01(values[frameOffset + 2 * RasterHeight * RasterWidth + cell]));

                        var targetX = (frameIndex * RasterWidth + x) * scale;
                        var targetY = outputHeight - ((y + 1) * scale);
                        for (var py = 0; py < scale; py++)
                        {
                            for (var px = 0; px < scale; px++)
                            {
                                texture.SetPixel(targetX + px, targetY + py, color);
                            }
                        }
                    }
                }
            }

            texture.Apply();
            return texture;
        }

        private float[] EncodeSemanticRgbFrameStack(
            ArenaSide perspective,
            float[] semanticChannels,
            ArenaObservationSnapshot observation)
        {
            var currentFrame = ConvertSemanticChannelsToRgb(semanticChannels, observation);
            var stack = GetRgbFrameStack(perspective);
            if (stack.Count == 0)
            {
                for (var i = 0; i < FrameStackSize; i++)
                {
                    stack.Add(CloneFrame(currentFrame));
                }
            }
            else
            {
                stack.Add(currentFrame);
                while (stack.Count > FrameStackSize)
                {
                    stack.RemoveAt(0);
                }

                while (stack.Count < FrameStackSize)
                {
                    stack.Insert(0, CloneFrame(stack[0]));
                }
            }

            var output = new float[FrameStackChannelCount * RasterHeight * RasterWidth];
            var frameLength = RgbChannelCount * RasterHeight * RasterWidth;
            for (var i = 0; i < FrameStackSize; i++)
            {
                Array.Copy(stack[i], 0, output, i * frameLength, frameLength);
            }

            return output;
        }

        private List<float[]> GetRgbFrameStack(ArenaSide perspective)
        {
            return perspective == ArenaSide.Left ? leftRgbFrameStack : rightRgbFrameStack;
        }

        private static float[] CloneFrame(float[] frame)
        {
            var copy = new float[frame.Length];
            Array.Copy(frame, copy, frame.Length);
            return copy;
        }

        private float[] ConvertSemanticChannelsToRgb(float[] semanticChannels, ArenaObservationSnapshot observation)
        {
            var rgb = new float[RgbChannelCount * RasterHeight * RasterWidth];
            var selfHasItem = semanticChannels[Index(10, 0, 0)] > 0.5f;
            var opponentHasItem = semanticChannels[Index(11, 0, 0)] > 0.5f;

            for (var y = 0; y < RasterHeight; y++)
            {
                for (var x = 0; x < RasterWidth; x++)
                {
                    var cell = y * RasterWidth + x;
                    var r = 0f;
                    var g = 0f;
                    var b = 0f;

                    if (semanticChannels[Index(5, y, x)] > 0.5f)
                    {
                        r = 0.08f;
                        g = 0.08f;
                        b = 0.08f;
                    }

                    if (semanticChannels[Index(4, y, x)] > 0.5f)
                    {
                        r = 0.26f;
                        g = 0.26f;
                        b = 0.26f;
                    }

                    if (semanticChannels[Index(6, y, x)] > 0.5f)
                    {
                        r = 0.95f;
                        g = 0.45f;
                        b = 0.08f;
                    }

                    var danger = semanticChannels[Index(9, y, x)];
                    if (danger > 0f)
                    {
                        r = Mathf.Max(r, 0.55f + 0.35f * danger);
                        g *= 1f - 0.45f * danger;
                        b *= 1f - 0.45f * danger;
                    }

                    var switchValue = semanticChannels[Index(7, y, x)];
                    if (switchValue > 0f)
                    {
                        r = Mathf.Max(r, 0.05f);
                        g = Mathf.Max(g, 0.55f + 0.4f * switchValue);
                        b = Mathf.Max(b, 0.7f);
                    }

                    if (semanticChannels[Index(8, y, x)] > 0.5f)
                    {
                        r = 0.9f;
                        g = 0.05f;
                        b = 1f;
                    }

                    if (semanticChannels[Index(3, y, x)] > 0.5f)
                    {
                        r = 0.1f;
                        g = 0.95f;
                        b = 0.25f;
                    }

                    if (semanticChannels[Index(2, y, x)] > 0.5f)
                    {
                        r = 1f;
                        g = 0.9f;
                        b = 0.05f;
                    }

                    if (semanticChannels[Index(1, y, x)] > 0.5f)
                    {
                        r = 1f;
                        g = opponentHasItem ? 0.55f : 0.06f;
                        b = opponentHasItem ? 0.04f : 0.1f;
                    }

                    if (semanticChannels[Index(0, y, x)] > 0.5f)
                    {
                        r = selfHasItem ? 0.05f : 0.08f;
                        g = selfHasItem ? 0.95f : 0.45f;
                        b = 1f;
                    }

                    rgb[cell] = Mathf.Clamp01(r);
                    rgb[RasterHeight * RasterWidth + cell] = Mathf.Clamp01(g);
                    rgb[2 * RasterHeight * RasterWidth + cell] = Mathf.Clamp01(b);
                }
            }

            WriteStatusMarker(rgb, selfHasItem, opponentHasItem);
            if (drawTargetHighlight)
            {
                WriteTargetHighlight(rgb, observation.targetPosition, observation.targetType);
            }

            return rgb;
        }

        private void WriteTargetHighlight(float[] rgb, Vector2 targetPosition, ArenaTargetType targetType)
        {
            var center = WorldToCell(targetPosition);
            var radius = Mathf.Max(1, targetHighlightRadiusCells);
            var color = targetType switch
            {
                ArenaTargetType.Base => new Vector3(0.05f, 1f, 0.15f),
                ArenaTargetType.Opponent => new Vector3(1f, 0.15f, 0.05f),
                _ => new Vector3(1f, 0.95f, 0.05f)
            };

            for (var y = center.y - radius; y <= center.y + radius; y++)
            {
                for (var x = center.x - radius; x <= center.x + radius; x++)
                {
                    if (!IsInside(y, x))
                    {
                        continue;
                    }

                    var manhattan = Mathf.Abs(y - center.y) + Mathf.Abs(x - center.x);
                    if (manhattan > radius + 1)
                    {
                        continue;
                    }

                    WriteRgbCell(rgb, y, x, color.x, color.y, color.z);
                }
            }
        }

        private static void WriteStatusMarker(float[] rgb, bool selfHasItem, bool opponentHasItem)
        {
            if (selfHasItem)
            {
                for (var y = RasterHeight - 3; y < RasterHeight; y++)
                {
                    for (var x = 0; x < 6; x++)
                    {
                        WriteRgbCell(rgb, y, x, 0.05f, 0.95f, 1f);
                    }
                }
            }

            if (opponentHasItem)
            {
                for (var y = RasterHeight - 3; y < RasterHeight; y++)
                {
                    for (var x = RasterWidth - 6; x < RasterWidth; x++)
                    {
                        WriteRgbCell(rgb, y, x, 1f, 0.55f, 0.04f);
                    }
                }
            }
        }

        private static void WriteRgbCell(float[] rgb, int y, int x, float r, float g, float b)
        {
            if (!IsInside(y, x))
            {
                return;
            }

            var cell = y * RasterWidth + x;
            rgb[cell] = r;
            rgb[RasterHeight * RasterWidth + cell] = g;
            rgb[2 * RasterHeight * RasterWidth + cell] = b;
        }

        public void RefreshStaticObstacleCache()
        {
            staticObstacleCache.Clear();
            var root = arenaRoot != null ? arenaRoot : transform.parent;
            if (root == null)
            {
                return;
            }

            foreach (var collider in root.GetComponentsInChildren<BoxCollider2D>(includeInactive: true))
            {
                if (collider == null
                    || collider.isTrigger
                    || collider.GetComponent<ArenaDoor>() != null
                    || collider.GetComponent<ArenaSwitch>() != null
                    || collider.GetComponent<ArenaMovingObstacle>() != null
                    || collider.GetComponent<ArenaItem>() != null
                    || collider.GetComponent<ArenaBaseZone>() != null
                    || collider.GetComponent<ArenaCharacterController>() != null)
                {
                    continue;
                }

                staticObstacleCache.Add(collider);
            }
        }

        private void WriteStaticObstacles(float[] values, int channel)
        {
            foreach (var obstacle in staticObstacleCache)
            {
                WriteColliderBounds(values, channel, obstacle.bounds, 1f);
            }
        }

        private void WriteWalkable(float[] values, int channel)
        {
            FillChannel(values, channel, 1f);
            foreach (var obstacle in staticObstacleCache)
            {
                WriteColliderBounds(values, channel, obstacle.bounds, 0f);
            }
        }

        private void WriteDoors(float[] values, int channel)
        {
            var doors = manager != null ? manager.ArenaDoors : null;
            if (doors == null)
            {
                return;
            }

            foreach (var door in doors)
            {
                if (door == null || door.IsOpen)
                {
                    continue;
                }

                var collider = door.GetComponent<BoxCollider2D>();
                if (collider != null)
                {
                    WriteColliderBounds(values, channel, collider.bounds, 1f);
                }
            }
        }

        private void WriteSwitches(float[] values, int channel)
        {
            var switches = manager != null ? manager.ArenaSwitches : null;
            if (switches == null)
            {
                return;
            }

            foreach (var arenaSwitch in switches)
            {
                if (arenaSwitch != null)
                {
                    WritePoint(values, channel, arenaSwitch.transform.position, arenaSwitch.IsActive ? 1f : 0.65f);
                }
            }
        }

        private void WriteMovingObstacles(float[] values, int channel)
        {
            var obstacles = manager != null ? manager.MovingObstacles : null;
            if (obstacles == null)
            {
                return;
            }

            foreach (var obstacle in obstacles)
            {
                if (obstacle == null)
                {
                    continue;
                }

                var collider = obstacle.GetComponent<BoxCollider2D>();
                if (collider != null)
                {
                    WriteColliderBounds(values, channel, collider.bounds, 1f);
                }
                else
                {
                    WritePoint(values, channel, obstacle.transform.position, 1f);
                }
            }
        }

        private void WriteMovingObstacleDanger(float[] values, int channel)
        {
            var obstacles = manager != null ? manager.MovingObstacles : null;
            if (obstacles == null)
            {
                return;
            }

            foreach (var obstacle in obstacles)
            {
                if (obstacle != null)
                {
                    WriteDanger(values, channel, obstacle.transform.position, movingObstacleDangerRadius);
                }
            }
        }

        private void WriteColliderBounds(float[] values, int channel, Bounds bounds, float value)
        {
            var min = WorldToCell(bounds.min);
            var max = WorldToCell(bounds.max);
            for (var y = Mathf.Min(min.y, max.y); y <= Mathf.Max(min.y, max.y); y++)
            {
                for (var x = Mathf.Min(min.x, max.x); x <= Mathf.Max(min.x, max.x); x++)
                {
                    WriteCell(values, channel, y, x, value);
                }
            }
        }

        private void WritePoint(float[] values, int channel, Vector2 position, float value)
        {
            var cell = WorldToCell(position);
            WriteCell(values, channel, cell.y, cell.x, value);
        }

        private void WriteDanger(float[] values, int channel, Vector2 center, float radius)
        {
            var centerCell = WorldToCell(center);
            var searchX = Mathf.CeilToInt(radius / Mathf.Max(CellWorldSize.x, 0.0001f));
            var searchY = Mathf.CeilToInt(radius / Mathf.Max(CellWorldSize.y, 0.0001f));

            for (var y = centerCell.y - searchY; y <= centerCell.y + searchY; y++)
            {
                for (var x = centerCell.x - searchX; x <= centerCell.x + searchX; x++)
                {
                    if (!IsInside(y, x))
                    {
                        continue;
                    }

                    var distance = Vector2.Distance(CellCenterWorld(y, x), center);
                    if (distance <= radius)
                    {
                        var index = Index(channel, y, x);
                        values[index] = Mathf.Max(values[index], 1f - distance / Mathf.Max(radius, 0.0001f));
                    }
                }
            }
        }

        private Vector2Int WorldToCell(Vector2 position)
        {
            var normalized = new Vector2(
                Mathf.InverseLerp(worldMin.x, worldMax.x, position.x),
                Mathf.InverseLerp(worldMin.y, worldMax.y, position.y));
            var x = Mathf.Clamp(Mathf.FloorToInt(normalized.x * RasterWidth), 0, RasterWidth - 1);
            var y = Mathf.Clamp(Mathf.FloorToInt((1f - normalized.y) * RasterHeight), 0, RasterHeight - 1);
            return new Vector2Int(x, y);
        }

        private Vector2 CellCenterWorld(int y, int x)
        {
            return new Vector2(
                Mathf.Lerp(worldMin.x, worldMax.x, (x + 0.5f) / RasterWidth),
                Mathf.Lerp(worldMin.y, worldMax.y, 1f - (y + 0.5f) / RasterHeight));
        }

        private Vector2 CellWorldSize => new(
            (worldMax.x - worldMin.x) / RasterWidth,
            (worldMax.y - worldMin.y) / RasterHeight);

        private static void FillChannel(float[] values, int channel, float value)
        {
            var offset = channel * RasterHeight * RasterWidth;
            for (var i = 0; i < RasterHeight * RasterWidth; i++)
            {
                values[offset + i] = value;
            }
        }

        private static void WriteCell(float[] values, int channel, int y, int x, float value)
        {
            if (!IsInside(y, x))
            {
                return;
            }

            values[Index(channel, y, x)] = value;
        }

        private static int Index(int channel, int y, int x)
        {
            return channel * RasterHeight * RasterWidth + y * RasterWidth + x;
        }

        private static bool IsInside(int y, int x)
        {
            return x >= 0 && y >= 0 && x < RasterWidth && y < RasterHeight;
        }

        private void LateUpdate()
        {
            if (!drawDebugOverlay)
            {
                ClearOverlay();
                return;
            }

            DrawOverlay();
        }

        private void DrawOverlay()
        {
            ClearOverlay();
            var values = EncodeSemanticChannelsFlat(ArenaSide.Left);
            var channel = 9;
            for (var y = 0; y < RasterHeight; y++)
            {
                for (var x = 0; x < RasterWidth; x++)
                {
                    var value = values[Index(channel, y, x)];
                    if (value <= 0.05f)
                    {
                        continue;
                    }

                    var cell = GameObject.CreatePrimitive(PrimitiveType.Quad);
                    cell.name = $"Raster Danger {x},{y}";
                    cell.transform.SetParent(transform, false);
                    cell.transform.position = CellCenterWorld(y, x);
                    cell.transform.localScale = new Vector3(CellWorldSize.x * 0.92f, CellWorldSize.y * 0.92f, 1f);
                    var renderer = cell.GetComponent<MeshRenderer>();
                    renderer.sharedMaterial = new Material(Shader.Find("Sprites/Default"))
                    {
                        color = new Color(1f, 0.2f, 0.1f, 0.18f * value)
                    };
                    var collider = cell.GetComponent<Collider>();
                    if (collider != null)
                    {
                        Destroy(collider);
                    }
                    overlayCells.Add(cell);
                }
            }
        }

        private void ClearOverlay()
        {
            foreach (var cell in overlayCells)
            {
                if (cell != null)
                {
                    Destroy(cell);
                }
            }

            overlayCells.Clear();
        }
    }
}
