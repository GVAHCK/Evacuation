"""Script to generate production-quality branded assets for the React Native/Expo mobile app.
Produces:
  - mobile/assets/icon.png (1024x1024)
  - mobile/assets/splash.png (1284x2778)
  - mobile/assets/adaptive-icon.png (1024x1024)
  - mobile/assets/favicon.png (48x48)
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_branding_assets():
    assets_dir = Path(__file__).resolve().parent.parent / "mobile" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    bg_color = (10, 15, 29)       # #0A0F1D
    cyan = (0, 229, 255)          # #00E5FF
    deep_blue = (26, 41, 78)      # #1A294E
    gold = (255, 179, 0)          # #FFB300
    white = (255, 255, 255)
    soft_cyan = (128, 240, 255)

    def draw_shield_crest(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float):
        """Draws the Evacuation Intelligence shield crest with network routing nodes."""
        w = int(220 * scale)
        h = int(270 * scale)
        
        # Shield polygon points
        points = [
            (cx - w, cy - h // 2),
            (cx + w, cy - h // 2),
            (cx + w, cy + h // 6),
            (cx, cy + h),
            (cx - w, cy + h // 6),
        ]
        
        # Outer glow/border
        draw.polygon(points, fill=deep_blue, outline=cyan, width=max(4, int(8 * scale)))
        
        # Inner shield
        inner_w = int(w * 0.85)
        inner_h = int(h * 0.85)
        inner_points = [
            (cx - inner_w, cy - inner_h // 2),
            (cx + inner_w, cy - inner_h // 2),
            (cx + inner_w, cy + inner_h // 6),
            (cx, cy + inner_h),
            (cx - inner_w, cy + inner_h // 6),
        ]
        draw.polygon(inner_points, fill=(15, 23, 42), outline=soft_cyan, width=max(2, int(4 * scale)))
        
        # Draw dynamic route network lines inside shield
        n1 = (cx - int(70 * scale), cy - int(40 * scale))
        n2 = (cx + int(60 * scale), cy - int(60 * scale))
        n3 = (cx, cy + int(10 * scale))
        n4 = (cx - int(50 * scale), cy + int(90 * scale))
        n5 = (cx + int(50 * scale), cy + int(80 * scale))
        
        lines = [(n1, n2), (n1, n3), (n2, n3), (n3, n4), (n3, n5), (n4, n5)]
        for p1, p2 in lines:
            draw.line([p1, p2], fill=cyan, width=max(2, int(4 * scale)))
            
        # Draw nodes
        r = max(4, int(9 * scale))
        for pt in [n1, n2, n3, n4]:
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=gold, outline=white, width=max(1, int(2 * scale)))
            
        # Destination node
        r_dest = max(6, int(13 * scale))
        draw.ellipse([n5[0] - r_dest, n5[1] - r_dest, n5[0] + r_dest, n5[1] + r_dest], fill=cyan, outline=white, width=max(2, int(3 * scale)))

    # 1. icon.png (1024 x 1024)
    print("Generating icon.png (1024x1024)...")
    img_icon = Image.new("RGBA", (1024, 1024), bg_color + (255,))
    draw_icon = ImageDraw.Draw(img_icon)
    draw_shield_crest(draw_icon, 512, 512, 1.8)
    img_icon.save(assets_dir / "icon.png", format="PNG")

    # 2. adaptive-icon.png (1024 x 1024)
    print("Generating adaptive-icon.png (1024x1024)...")
    img_adaptive = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    draw_adaptive = ImageDraw.Draw(img_adaptive)
    # Circle container in center
    draw_adaptive.ellipse([212, 212, 812, 812], fill=(15, 23, 42, 255), outline=cyan, width=8)
    draw_shield_crest(draw_adaptive, 512, 512, 1.3)
    img_adaptive.save(assets_dir / "adaptive-icon.png", format="PNG")

    # 3. splash.png (1284 x 2778)
    print("Generating splash.png (1284x2778)...")
    img_splash = Image.new("RGBA", (1284, 2778), bg_color + (255,))
    draw_splash = ImageDraw.Draw(img_splash)
    draw_shield_crest(draw_splash, 642, 1200, 2.0)
    img_splash.save(assets_dir / "splash.png", format="PNG")

    # 4. favicon.png (48 x 48)
    print("Generating favicon.png (48x48)...")
    img_favicon = Image.new("RGBA", (48, 48), bg_color + (255,))
    draw_fav = ImageDraw.Draw(img_favicon)
    draw_shield_crest(draw_fav, 24, 24, 0.08)
    img_favicon.save(assets_dir / "favicon.png", format="PNG")

    print(f"All mobile branding assets successfully generated in {assets_dir}")

if __name__ == "__main__":
    create_branding_assets()
