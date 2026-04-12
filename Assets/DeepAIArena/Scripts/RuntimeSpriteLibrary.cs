using UnityEngine;

namespace DeepAIArena
{
    public static class RuntimeSpriteLibrary
    {
        private static Sprite square;

        public static Sprite Square
        {
            get
            {
                if (square != null)
                {
                    return square;
                }

                var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
                texture.SetPixel(0, 0, Color.white);
                texture.Apply();
                texture.filterMode = FilterMode.Point;

                square = Sprite.Create(texture, new Rect(0f, 0f, 1f, 1f), new Vector2(0.5f, 0.5f), 1f);
                square.name = "RuntimeSquare";
                return square;
            }
        }
    }
}
