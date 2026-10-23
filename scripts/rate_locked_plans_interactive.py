"""
Interactive Rating Tool for the 12 SHA256-Locked External Floorplans.
Allows each teammate (Khalil, Amine, Khawla) to independently rate the 12 plans
as 'Usable', 'Needs Fixes', or 'Unusable'.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "real_external_plans" / "fresh_defense_eval" / "fresh_defense_manifest.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "fresh_defense_results" / "fresh_defense_evaluation_report.json"
OUTPUT_RATINGS = PROJECT_ROOT / "data" / "real_external_plans" / "fresh_defense_eval" / "human_team_ratings.json"

CATEGORIES = {
    "1": ("Usable", "Directement exploitable pour l'extrusion 3D, agencement fidele, retouches minimes."),
    "2": ("Needs Fixes", "Partiellement exploitable, necessite des retouches manuelles CAD significatives."),
    "3": ("Unusable", "Inexploitable / echec total (0 piece, coque ouverte, ou geometrie corrompue).")
}

def rate_plans(annotator_name: str):
    if not REPORT_PATH.exists():
        print(f"Error: Evaluation report not found at {REPORT_PATH}")
        return

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    ratings = {}
    print("=" * 80)
    print(f"Human Architectural Evaluation for: {annotator_name}")
    print("Categories:")
    for k, (cat, desc) in CATEGORIES.items():
        print(f"  [{k}] {cat.upper()}: {desc}")
    print("=" * 80)

    for p in eval_data["plans"]:
        pid = p["id"]
        style = p["style"]
        pr = p.get("pred_rooms", 0)
        gr = p["gt_rooms"]
        pd = p.get("pred_doors", 0)
        gd = p["gt_doors"]
        shell = p.get("shell_status", "UNKNOWN")
        print(f"\nPlan {pid}: {style}")
        print(f"  Rooms: {pr} (GT: {gr}) | Doors: {pd} (GT: {gd}) | Shell: {shell}")

        while True:
            choice = input("  Rating (1: Usable, 2: Needs Fixes, 3: Unusable): ").strip()
            if choice in CATEGORIES:
                ratings[pid] = CATEGORIES[choice][0]
                break
            print("  Invalid choice. Enter 1, 2, or 3.")

    # Save to disk
    all_ratings = {}
    if OUTPUT_RATINGS.exists():
        with open(OUTPUT_RATINGS, "r", encoding="utf-8") as f:
            all_ratings = json.load(f)

    all_ratings[annotator_name] = ratings
    with open(OUTPUT_RATINGS, "w", encoding="utf-8") as f:
        json.dump(all_ratings, f, indent=2)

    print(f"\nRatings saved for {annotator_name} to {OUTPUT_RATINGS}")

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else input("Enter your name (Khalil / Amine / Khawla): ").strip()
    rate_plans(name)
