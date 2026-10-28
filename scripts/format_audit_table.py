import json
from pathlib import Path

def generate_markdown_table():
    with open("outputs/blind_study/data/selection_audit_full_table.json") as f:
        data = json.load(f)

    rooms = data["rooms"]
    status_counts = data["status_counts"]

    lines = []
    lines.append("| Plan Index | Plan Filename | Room ID | Room Type | Area (m²) | Status | Details / Exclusion Reason |")
    lines.append("|:---:|:---|:---:|:---|:---:|:---|:---|")

    for r in rooms:
        pidx = r["val_plan_index"]
        fname = f"`{r['plan_filename']}`"
        rid = r["room_index"]
        rtype = r["norm_type"].replace("_", " ").title()
        area = f"{r['area_m2']:.2f}"
        status = f"**{r['status']}**" if r["status"] == "selected" else r["status"]
        details = r["details"]
        lines.append(f"| {pidx} | {fname} | {rid} | {rtype} | {area} | {status} | {details} |")

    table_md = "\n".join(lines)
    with open("outputs/blind_study/data/selection_audit_table.md", "w", encoding="utf-8") as f:
        f.write(table_md)
    print("Generated selection_audit_table.md successfully!")

if __name__ == "__main__":
    generate_markdown_table()
