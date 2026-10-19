"""Verify Mobile Assets Script.
Inspects mobile/app.json and asserts every referenced local asset file actually exists.
"""

import os
import sys
import json
from pathlib import Path

def verify_mobile_assets() -> bool:
    print("=" * 60)
    print("VERIFYING MOBILE APP.JSON ASSET EXISTENCE")
    print("=" * 60)

    repo_root = Path(__file__).resolve().parent.parent
    app_json_path = repo_root / "mobile" / "app.json"

    if not app_json_path.exists():
        print(f"[FAIL] app.json not found at {app_json_path}")
        return False

    with open(app_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expo_cfg = data.get("expo", {})
    referenced_assets = []

    # 1. Top-level icon
    if "icon" in expo_cfg:
        referenced_assets.append(("icon", expo_cfg["icon"]))

    # 2. Splash image
    splash = expo_cfg.get("splash", {})
    if "image" in splash:
        referenced_assets.append(("splash.image", splash["image"]))

    # 3. Android adaptiveIcon
    android_adaptive = expo_cfg.get("android", {}).get("adaptiveIcon", {})
    if "foregroundImage" in android_adaptive:
        referenced_assets.append(("android.adaptiveIcon.foregroundImage", android_adaptive["foregroundImage"]))
    if "backgroundImage" in android_adaptive:
        referenced_assets.append(("android.adaptiveIcon.backgroundImage", android_adaptive["backgroundImage"]))

    # 4. Web favicon
    web = expo_cfg.get("web", {})
    if "favicon" in web:
        referenced_assets.append(("web.favicon", web["favicon"]))

    # 5. Notification icon
    notification = expo_cfg.get("notification", {})
    if "icon" in notification:
        referenced_assets.append(("notification.icon", notification["icon"]))

    mobile_dir = repo_root / "mobile"
    all_exist = True

    for key, rel_path in referenced_assets:
        target_path = (mobile_dir / rel_path).resolve()
        if not target_path.exists():
            print(f"[FAIL] {key:<35} -> Missing: {rel_path} ({target_path})")
            all_exist = False
        else:
            size_kb = target_path.stat().st_size / 1024.0
            print(f"[PASS] {key:<35} -> Exists:  {rel_path} ({size_kb:.1f} KB)")

    print("-" * 60)
    if all_exist:
        print("ALL REFERENCED MOBILE ASSETS EXIST AND ARE VALID!")
    else:
        print("MOBILE ASSET VERIFICATION FAILED! Missing assets detected.")
    print("-" * 60)
    return all_exist

if __name__ == "__main__":
    success = verify_mobile_assets()
    sys.exit(0 if success else 1)
