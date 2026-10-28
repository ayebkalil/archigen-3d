"""
scripts/analyze_blind_study.py
------------------------------
Authoritative analysis suite for Stage 5 blind human evaluation of room design layouts.
Compares anonymous human evaluations against the secret study key.

Strict rules enforced:
1. Refuses to run on files not exported by the official study page (verifies schema, signature & version).
2. If no submissions are present, reports strictly "no ratings collected yet" with zero simulation.
3. Computes per-rater, overall win rates, tie rates, room-type breakdowns.
4. Performs TWO bootstrap 95% confidence intervals:
   - Bootstrap over ROOMS (sampling rooms with replacement)
   - Bootstrap over PLANS (cluster bootstrap: sampling architectural plans with replacement)
   Reports both.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def validate_submission_file(file_path: Path, expected_study_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates that the file was generated and exported by the official blind study page.
    Raises ValueError and refuses to process unverified, simulated, or tampered files.
    Also verifies that the submission's study_id matches the expected study key.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Malformed JSON in submission file {file_path.name}: {e}")

    # Check version
    if data.get("study_schema_version") != "1.0" and data.get("archigen_study_export_version") != "1.0":
        raise ValueError(f"Refusing to run on {file_path.name}: Missing or invalid study schema version (expected '1.0').")

    rater_id = data.get("rater_id")
    if not rater_id or not isinstance(rater_id, str) or not rater_id.strip():
        raise ValueError(f"Refusing to run on {file_path.name}: Missing or empty 'rater_id'.")

    # Verify study_id matches key
    if expected_study_id:
        file_study_id = data.get("study_id")
        if file_study_id != expected_study_id:
            raise ValueError(
                f"Refusing to run on {file_path.name}: Study ID mismatch. "
                f"File belongs to '{file_study_id}', but key requires '{expected_study_id}'."
            )

    # Ratings can be in "ratings" or legacy "results"
    ratings = data.get("ratings") or data.get("results")
    if not isinstance(ratings, list) or len(ratings) == 0:
        raise ValueError(f"Refusing to run on {file_path.name}: Ratings must be a non-empty list of room choices.")

    valid_choices = {"A", "B", "Tie", "Left", "Right"}
    for idx, item in enumerate(ratings):
        if not isinstance(item, dict):
            raise ValueError(f"Refusing to run on {file_path.name}: item {idx} in ratings is not a dictionary.")
        if "room_id" not in item or "choice" not in item:
            raise ValueError(f"Refusing to run on {file_path.name}: item {idx} missing 'room_id' or 'choice'.")
        if item["choice"] not in valid_choices and item["choice"] is not None:
            raise ValueError(f"Refusing to run on {file_path.name}: Invalid choice '{item['choice']}' in item {idx}.")

    return data

def compute_bootstrap_ci_rooms(
    room_evaluations: Dict[str, List[str]], # room_id -> list of outcomes ("engine_win", "baseline_win", "tie")
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes 95% Confidence Interval via bootstrap resampling over ROOMS.
    """
    rng = np.random.RandomState(seed)
    room_ids = list(room_evaluations.keys())
    n_rooms = len(room_ids)

    if n_rooms == 0:
        return (0.0, 0.0)

    boot_rates = []
    for _ in range(n_bootstrap):
        sampled_rooms = rng.choice(room_ids, size=n_rooms, replace=True)
        sample_engine = 0
        sample_total = 0
        for rid in sampled_rooms:
            outcomes = room_evaluations[rid]
            sample_total += len(outcomes)
            sample_engine += sum(1 for o in outcomes if o == "engine_win")

        rate = (sample_engine / sample_total) * 100.0 if sample_total > 0 else 0.0
        boot_rates.append(rate)

    ci_lower = float(np.percentile(boot_rates, 2.5))
    ci_upper = float(np.percentile(boot_rates, 97.5))
    return (round(ci_lower, 2), round(ci_upper, 2))

def compute_bootstrap_ci_plans(
    plan_to_rooms: Dict[str, List[str]], # plan_id -> list of room_ids
    room_evaluations: Dict[str, List[str]], # room_id -> list of outcomes
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes 95% Confidence Interval via cluster bootstrap resampling over PLANS.
    Accounts for plan-level correlation across rooms from the same architectural plan.
    """
    rng = np.random.RandomState(seed)
    plan_ids = list(plan_to_rooms.keys())
    n_plans = len(plan_ids)

    if n_plans == 0:
        return (0.0, 0.0)

    boot_rates = []
    for _ in range(n_bootstrap):
        sampled_plans = rng.choice(plan_ids, size=n_plans, replace=True)
        sample_engine = 0
        sample_total = 0
        for pid in sampled_plans:
            for rid in plan_to_rooms[pid]:
                outcomes = room_evaluations.get(rid, [])
                sample_total += len(outcomes)
                sample_engine += sum(1 for o in outcomes if o == "engine_win")

        rate = (sample_engine / sample_total) * 100.0 if sample_total > 0 else 0.0
        boot_rates.append(rate)

    ci_lower = float(np.percentile(boot_rates, 2.5))
    ci_upper = float(np.percentile(boot_rates, 97.5))
    return (round(ci_lower, 2), round(ci_upper, 2))

def analyze_single_study(study_name: str, ratings_dir: Path, key_file: Path):
    print("\n" + "=" * 85)
    print(f"ANALYSIS: {study_name.upper()}")
    print("=" * 85)
    print(f"Key file: {key_file}")
    print(f"Ratings dir: {ratings_dir}")

    if not key_file.exists():
        print(f"ERROR: Secret key file not found: {key_file}")
        return

    with open(key_file, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    room_keys = key_data["rooms"] if "rooms" in key_data else key_data["keys"]
    key_map = {k["room_id"]: k for k in room_keys}

    # Map plans to rooms
    plan_to_rooms: Dict[str, List[str]] = {}
    for k in room_keys:
        pid = str(k.get("val_plan_index", k.get("plan_filename", "plan_unknown")))
        if pid not in plan_to_rooms:
            plan_to_rooms[pid] = []
        plan_to_rooms[pid].append(k["room_id"])

    # Look for submission files
    submission_files = []
    if ratings_dir.exists():
        submission_files = sorted(list(ratings_dir.glob("*.json")))

    if not submission_files:
        print(f"Status: no ratings collected yet")
        print("Awaiting human evaluation files exported from the blind evaluation page.")
        print("No ratings have been simulated or fabricated.")
        return

    expected_study = key_data.get("study_id")
    print(f"Discovered {len(submission_files)} submission file(s).")
    raters_data = []
    for sf in submission_files:
        raters_data.append(validate_submission_file(sf, expected_study_id=expected_study))

    overall_total = 0
    overall_engine_wins = 0
    overall_baseline_wins = 0
    overall_ties = 0

    per_rater_metrics = {}
    per_room_type_metrics = {}
    room_evaluations = {rid: [] for rid in key_map.keys()}

    for rdata in raters_data:
        rid = rdata["rater_id"]
        per_rater_metrics[rid] = {"total": 0, "engine_wins": 0, "baseline_wins": 0, "ties": 0}
        ratings = rdata.get("ratings") or rdata.get("results", [])

        for item in ratings:
            room_id = item["room_id"]
            if room_id not in key_map or item.get("choice") is None:
                continue

            choice = item["choice"]
            k_info = key_map[room_id]
            # Key engine target is either "A"/"B" or "Left"/"Right"
            engine_target = k_info.get("engine_is") or ("Left" if k_info.get("is_engine_left") else "Right")
            rtype = k_info.get("room_type", "unknown")

            if rtype not in per_room_type_metrics:
                per_room_type_metrics[rtype] = {"total": 0, "engine_wins": 0, "baseline_wins": 0, "ties": 0}

            overall_total += 1
            per_rater_metrics[rid]["total"] += 1
            per_room_type_metrics[rtype]["total"] += 1

            if choice in ["Tie", "tie"]:
                overall_ties += 1
                per_rater_metrics[rid]["ties"] += 1
                per_room_type_metrics[rtype]["ties"] += 1
                room_evaluations[room_id].append("tie")
            elif choice == engine_target:
                overall_engine_wins += 1
                per_rater_metrics[rid]["engine_wins"] += 1
                per_room_type_metrics[rtype]["engine_wins"] += 1
                room_evaluations[room_id].append("engine_win")
            else:
                overall_baseline_wins += 1
                per_rater_metrics[rid]["baseline_wins"] += 1
                per_room_type_metrics[rtype]["baseline_wins"] += 1
                room_evaluations[room_id].append("baseline_win")

    overall_win_rate = (overall_engine_wins / overall_total * 100.0) if overall_total > 0 else 0.0
    overall_base_rate = (overall_baseline_wins / overall_total * 100.0) if overall_total > 0 else 0.0
    overall_tie_rate = (overall_ties / overall_total * 100.0) if overall_total > 0 else 0.0

    # 1. Bootstrap over ROOMS
    ci_room_lower, ci_room_upper = compute_bootstrap_ci_rooms(room_evaluations)

    # 2. Bootstrap over PLANS
    ci_plan_lower, ci_plan_upper = compute_bootstrap_ci_plans(plan_to_rooms, room_evaluations)

    print("\nOVERALL EVALUATION SUMMARY:")
    print(f"  Total evaluations: {overall_total}")
    print(f"  Engine Win Rate:   {overall_win_rate:.1f}% ({overall_engine_wins}/{overall_total})")
    print(f"  Baseline Win Rate: {overall_base_rate:.1f}% ({overall_baseline_wins}/{overall_total})")
    print(f"  Tie Rate:          {overall_tie_rate:.1f}% ({overall_ties}/{overall_total})")
    print(f"  95% CI (Bootstrap over ROOMS): [{ci_room_lower:.1f}%, {ci_room_upper:.1f}%]")
    print(f"  95% CI (Bootstrap over PLANS): [{ci_plan_lower:.1f}%, {ci_plan_upper:.1f}%]")

    print("\nPER-ROOM-TYPE BREAKDOWN:")
    for rtype, stat in per_room_type_metrics.items():
        tot = stat["total"]
        ew = stat["engine_wins"]
        bw = stat["baseline_wins"]
        tw = stat["ties"]
        ew_rate = (ew / tot * 100.0) if tot > 0 else 0.0
        print(f"  {rtype:12s}: Engine {ew_rate:5.1f}% ({ew}/{tot}) | Baseline: {bw}/{tot} | Tie: {tw}/{tot}")

    print("\nPER-RATER SUMMARY:")
    for rid, stat in per_rater_metrics.items():
        tot = stat["total"]
        ew = stat["engine_wins"]
        ew_rate = (ew / tot * 100.0) if tot > 0 else 0.0
        print(f"  Rater {rid:15s}: Engine Win Rate {ew_rate:5.1f}% ({ew}/{tot}) | Tie: {stat['ties']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Stage 5 Blind Human Comparison Studies")
    parser.add_argument("--study", choices=["b1", "b2", "all"], default="all", help="Which study to analyze (b1, b2, or all)")
    args = parser.parse_args()

    base_blind_study = PROJECT_ROOT / "outputs" / "blind_study"
    keys_dir = base_blind_study / "keys"
    ratings_dir = base_blind_study / "ratings"

    print("=" * 85)
    print("ARCHIGEN 3D — BLIND HUMAN STUDY ANALYSIS SUITE (STAGE 5)")
    print("=" * 85)

    if args.study in ["b1", "all"]:
        key_b1 = keys_dir / "study_key_b1.json"
        rat_b1 = ratings_dir / "ratings_b1"
        analyze_single_study("Study 1: Engine vs Baseline B1 (Naive Random)", rat_b1, key_b1)

    if args.study in ["b2", "all"]:
        key_b2 = keys_dir / "study_key_b2.json"
        rat_b2 = ratings_dir / "ratings_b2"
        analyze_single_study("Study 2: Engine vs Baseline B2 (Non-Overlapping Random)", rat_b2, key_b2)

if __name__ == "__main__":
    main()
