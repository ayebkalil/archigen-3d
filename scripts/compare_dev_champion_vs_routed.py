"""
Compare Champion v1.1 vs Full Routed System across all 20 Dev Plans.
Generates:
  1. Complete per-plan detection table (rooms, area, shell, model routed).
  2. Side-by-side visual overlays (outputs/dev_results/champion_vs_routed/).
  3. Interactive HTML rating app specifically pairing Champion vs Routed System for human evaluation.
"""

import sys
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

ARTIFACT_DIR = Path("C:/Users/ayebk/.gemini/antigravity/brain/387c7b87-863d-4c00-831b-93abdc64d4d4")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def render_overlay(img_path, layout, out_path):
    orig_img = Image.open(img_path).convert("RGB")
    w_orig, h_orig = orig_img.size
    scale = min(512 / w_orig, 512 / h_orig)
    nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
    pad_x = (512 - nw) // 2
    pad_y = (512 - nh) // 2

    canvas = Image.new("RGB", (512, 512), (255, 255, 255))
    canvas.paste(orig_img.resize((nw, nh), Image.Resampling.BILINEAR), (pad_x, pad_y))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Rooms
    scale_m = layout.get("metadata", {}).get("scale_m_per_px", 0.03)
    for r in layout.get("rooms", []):
        poly_m = r.get("polygon_meters", [])
        if len(poly_m) >= 3:
            pts_px = [(int(p[0] / scale_m), int(p[1] / scale_m)) for p in poly_m]
            draw.polygon(pts_px, fill=(60, 180, 75, 70), outline=(30, 120, 45, 200))

    # Walls
    for w in layout.get("walls", []):
        sp = w.get("start_point", [0, 0])
        ep = w.get("end_point", [0, 0])
        col = (255, 50, 50, 220) if w.get("is_exterior", False) else (50, 50, 255, 200)
        draw.line([(int(sp[0]), int(sp[1])), (int(ep[0]), int(ep[1]))], fill=col, width=2)

    # Doors
    for d in layout.get("doors", []):
        pos = d.get("position", [0, 0])
        draw.ellipse([(pos[0]-4, pos[1]-4), (pos[0]+4, pos[1]+4)], fill=(255, 200, 0, 240), outline=(0, 0, 0, 255))

    canvas.save(out_path)

def encode_b64(img_path):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((400, 400), Image.Resampling.BILINEAR)
        import io
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    manifest_p = PROJECT_ROOT / "data" / "dev_external_plans" / "dev_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_dir = PROJECT_ROOT / "outputs" / "dev_results" / "champion_vs_routed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FloorplanPipeline.get_instance()
    # Ensure calibrated threshold
    pipeline.vectorizer.min_area_habitable_m2 = 2.0

    print("=" * 110)
    print("DEV SET (N=20) PERFORMANCE COMPARISON: CHAMPION v1.1 vs ROUTED SYSTEM")
    print("=" * 110)
    print(f"{'Plan ID':<8} | {'Champion v1.1 (Rooms, Area)':<30} | {'Routed System (Rooms, Area)':<30} | {'Model Routed':<20} | {'Fallback?'}")
    print("-" * 110)

    records = []
    plan_html_data = []

    for item in manifest:
        pid = item["id"]
        fname = item["filename"]
        img_p = manifest_p.parent / fname

        with open(img_p, "rb") as f_img:
            raw = f_img.read()

        # 1. Run Champion v1.1 Standalone (disable fallback temporarily)
        fb_temp = pipeline.fallback_model
        pipeline.fallback_model = None
        res_champ = pipeline.process_image(raw, tta_mode="none", front_only=True)
        lay_champ = res_champ["layout"]
        n_c = len(lay_champ.get("rooms", []))
        area_c = lay_champ.get("metadata", {}).get("total_surface_m2", 0.0)
        valid_c = lay_champ.get("metadata", {}).get("is_building_shell_valid", False)

        # 2. Run Full Routed System (with fallback enabled)
        pipeline.fallback_model = fb_temp
        res_routed = pipeline.process_image(raw, tta_mode="none", front_only=True)
        lay_routed = res_routed["layout"]
        n_r = len(lay_routed.get("rooms", []))
        area_r = lay_routed.get("metadata", {}).get("total_surface_m2", 0.0)
        valid_r = lay_routed.get("metadata", {}).get("is_building_shell_valid", False)
        routing_meta = lay_routed.get("metadata", {}).get("routing", {})
        routed_model = routing_meta.get("model_used", "champion_v1_1")
        fb_triggered = routing_meta.get("fallback_triggered", False)

        # Save overlays
        p_champ_overlay = out_dir / f"{pid}_champion.png"
        p_routed_overlay = out_dir / f"{pid}_routed.png"
        render_overlay(img_p, lay_champ, p_champ_overlay)
        render_overlay(img_p, lay_routed, p_routed_overlay)

        c_str = f"{n_c:2d} rooms ({area_c:5.1f}m2, {'V' if valid_c else 'X'})"
        r_str = f"{n_r:2d} rooms ({area_r:5.1f}m2, {'V' if valid_r else 'X'})"
        fb_str = "YES (Triggered)" if fb_triggered else "NO (Clean)"
        print(f"{pid:<8} | {c_str:<30} | {r_str:<30} | {routed_model:<20} | {fb_str}")

        records.append({
            "id": pid,
            "filename": fname,
            "champion_rooms": n_c,
            "champion_area_m2": round(area_c, 1),
            "champion_valid": valid_c,
            "routed_rooms": n_r,
            "routed_area_m2": round(area_r, 1),
            "routed_valid": valid_r,
            "model_used": routed_model,
            "fallback_triggered": fb_triggered
        })

        plan_html_data.append({
            "id": pid,
            "filename": fname,
            "orig_b64": encode_b64(img_p),
            "champ_b64": encode_b64(p_champ_overlay),
            "routed_b64": encode_b64(p_routed_overlay),
            "champ_rooms": n_c,
            "champ_area": round(area_c, 1),
            "champ_valid": valid_c,
            "routed_rooms": n_r,
            "routed_area": round(area_r, 1),
            "routed_valid": valid_r,
            "fallback_triggered": fb_triggered,
            "model_used": routed_model
        })

    print("=" * 110)
    tot_c_rooms = sum(r["champion_rooms"] for r in records)
    tot_r_rooms = sum(r["routed_rooms"] for r in records)
    tot_fb = sum(1 for r in records if r["fallback_triggered"])
    print(f"Summary across 20 Dev Plans:")
    print(f"  Champion v1.1 Total Rooms: {tot_c_rooms} rooms (avg {tot_c_rooms/20:.2f} rooms/plan)")
    print(f"  Routed System Total Rooms: {tot_r_rooms} rooms (avg {tot_r_rooms/20:.2f} rooms/plan)")
    print(f"  Fallback Triggered:        {tot_fb} / 20 plans ({tot_fb/20*100:.1f}%)")
    print(f"  Valid Shells (Champion):   {sum(1 for r in records if r['champion_valid'])} / 20 ({sum(1 for r in records if r['champion_valid'])/20*100:.1f}%)")
    print(f"  Valid Shells (Routed):     {sum(1 for r in records if r['routed_valid'])} / 20 ({sum(1 for r in records if r['routed_valid'])/20*100:.1f}%)")
    print("=" * 110)

    # Save summary JSON
    out_json = PROJECT_ROOT / "outputs" / "dev_champion_vs_routed_summary.json"
    with open(out_json, "w") as f:
        json.dump({
            "total_plans": 20,
            "champion_total_rooms": tot_c_rooms,
            "routed_total_rooms": tot_r_rooms,
            "fallback_triggers": tot_fb,
            "records": records
        }, f, indent=2)
    print(f"[DEV EVAL] Summary saved to {out_json}")

    # Build Dedicated Champion vs Routed Side-by-Side Rater
    generate_champ_vs_routed_html(plan_html_data)

def generate_champ_vs_routed_html(plans):
    data_json = json.dumps(plans)
    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ArchiGen 3D - Champion v1.1 vs Routed System Rater</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .active-nav {{ border-color: #3b82f6; background-color: rgba(59, 130, 246, 0.1); color: #3b82f6; font-weight: 600; }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased min-h-screen p-4 md:p-6 font-sans">
  <header class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 mb-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
    <div>
      <div class="flex items-center gap-3">
        <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider rounded-md bg-blue-500/10 text-blue-500 border border-blue-500/20">Human Rater</span>
        <h1 class="text-xl md:text-2xl font-bold tracking-tight">Champion v1.1 vs Routed System (N=20 Dev Plans)</h1>
      </div>
      <p class="text-[var(--muted-foreground)] text-sm mt-1">
        Direct side-by-side human grading. Rate each plan independently as Usable, Needs Fixes, or Unusable.
      </p>
    </div>
    <div class="flex items-center gap-3">
      <div class="text-right">
        <div class="text-xs uppercase tracking-wider text-[var(--muted-foreground)]">Progress</div>
        <div class="text-base font-bold text-blue-500" id="progress-text">0 / 20 Rated</div>
      </div>
      <button onclick="exportRatings()" class="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-xl shadow transition">
        💾 Export Ratings JSON
      </button>
    </div>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
    <div class="lg:col-span-3 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-4 shadow-sm h-[780px] flex flex-col">
      <div class="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3 px-2 flex justify-between">
        <span>Plans (20)</span>
        <span id="sidebar-rated">0/20</span>
      </div>
      <div class="overflow-y-auto space-y-1.5 flex-1 pr-1" id="plan-list"></div>
    </div>

    <div class="lg:col-span-9 space-y-6">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-2">
            <span class="text-xs font-mono font-bold text-blue-500 px-2 py-0.5 rounded bg-blue-500/10" id="current-pid">dev_01</span>
            <span class="text-xs text-[var(--muted-foreground)]" id="current-fname"></span>
            <span id="badge-routed-tag" class="text-[10px] px-2 py-0.5 rounded font-bold"></span>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="navPlan(-1)" class="px-3 py-1.5 text-xs font-medium border border-[var(--border)] rounded-lg hover:bg-[var(--accent)] transition">‹ Prev</button>
            <button onclick="navPlan(1)" class="px-3 py-1.5 text-xs font-medium border border-[var(--border)] rounded-lg hover:bg-[var(--accent)] transition">Next ›</button>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div class="border border-[var(--border)] rounded-xl overflow-hidden bg-black/5 flex flex-col">
            <div class="px-3 py-2 bg-[var(--accent)] border-b border-[var(--border)] text-xs font-semibold">Original Blueprint</div>
            <div class="p-2 flex-1 flex items-center justify-center bg-white min-h-[280px]">
              <img id="img-orig" src="" class="max-h-[280px] max-w-full object-contain" />
            </div>
          </div>
          <div class="border border-[var(--border)] rounded-xl overflow-hidden bg-black/5 flex flex-col">
            <div class="px-3 py-2 bg-[var(--accent)] border-b border-[var(--border)] text-xs font-semibold flex justify-between">
              <span>Champion v1.1 (Production)</span>
              <span id="champ-stats" class="text-[10px] text-[var(--muted-foreground)]"></span>
            </div>
            <div class="p-2 flex-1 flex items-center justify-center bg-white min-h-[280px]">
              <img id="img-champ" src="" class="max-h-[280px] max-w-full object-contain" />
            </div>
          </div>
          <div class="border border-[var(--border)] rounded-xl overflow-hidden bg-black/5 flex flex-col">
            <div class="px-3 py-2 bg-[var(--accent)] border-b border-[var(--border)] text-xs font-semibold flex justify-between">
              <span>Routed System (Challenger Fallback)</span>
              <span id="routed-stats" class="text-[10px] text-blue-500 font-bold"></span>
            </div>
            <div class="p-2 flex-1 flex items-center justify-center bg-white min-h-[280px]">
              <img id="img-routed" src="" class="max-h-[280px] max-w-full object-contain" />
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-[var(--border)]">
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--card)] space-y-3">
            <h3 class="text-sm font-bold">Rate Champion v1.1</h3>
            <div class="grid grid-cols-3 gap-2">
              <button onclick="setScore('champ', 'usable')" id="btn-champ-usable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-emerald-500/10">✓ Usable (2)</button>
              <button onclick="setScore('champ', 'needs_fixes')" id="btn-champ-needs" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-amber-500/10">⚡ Needs Fixes (1)</button>
              <button onclick="setScore('champ', 'unusable')" id="btn-champ-unusable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-rose-500/10">✕ Unusable (0)</button>
            </div>
          </div>
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--card)] space-y-3">
            <h3 class="text-sm font-bold">Rate Routed System</h3>
            <div class="grid grid-cols-3 gap-2">
              <button onclick="setScore('routed', 'usable')" id="btn-routed-usable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-emerald-500/10">✓ Usable (2)</button>
              <button onclick="setScore('routed', 'needs_fixes')" id="btn-routed-needs" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-amber-500/10">⚡ Needs Fixes (1)</button>
              <button onclick="setScore('routed', 'unusable')" id="btn-routed-unusable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-rose-500/10">✕ Unusable (0)</button>
            </div>
          </div>
        </div>

        <div class="mt-4 pt-4 border-t border-[var(--border)] flex justify-between items-center">
          <div class="flex items-center gap-2">
            <span class="text-xs text-[var(--muted-foreground)]">Preference:</span>
            <button onclick="setPref('champ')" id="btn-pref-champ" class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-[var(--border)]">Prefer Champion</button>
            <button onclick="setPref('tie')" id="btn-pref-tie" class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-[var(--border)]">Tie / Equal</button>
            <button onclick="setPref('routed')" id="btn-pref-routed" class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-[var(--border)]">Prefer Routed</button>
          </div>
          <button onclick="navPlan(1)" class="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-xl shadow transition">Save & Next Plan →</button>
        </div>
      </div>

      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <h2 class="text-lg font-bold mb-3">Live Usability Scorecard (Human Rated)</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Champion v1.1 Usable</div>
            <div class="text-2xl font-bold mt-1" id="stat-champ">—</div>
          </div>
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Routed System Usable</div>
            <div class="text-2xl font-bold mt-1 text-blue-600" id="stat-routed">—</div>
          </div>
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Routed Head-to-Head Win</div>
            <div class="text-2xl font-bold mt-1 text-emerald-600" id="stat-win">—</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const plans = {data_json};
    let currentIndex = 0;
    let ratings = JSON.parse(localStorage.getItem('archigen_champ_vs_routed_ratings') || '{{}}');

    function init() {{
      renderSidebar();
      loadPlan(0);
      updateStats();
    }}

    function renderSidebar() {{
      const list = document.getElementById('plan-list');
      list.innerHTML = '';
      let rated = 0;
      plans.forEach((p, idx) => {{
        const r = ratings[p.id];
        const isR = r && r.champ && r.routed;
        if (isR) rated++;
        const b = document.createElement('button');
        b.className = `w-full text-left px-3 py-2 rounded-xl border text-xs flex justify-between items-center transition ${{idx === currentIndex ? 'active-nav' : 'border-[var(--border)] hover:bg-[var(--accent)]'}}`;
        b.onclick = () => loadPlan(idx);
        b.innerHTML = `<span class="font-mono font-bold">${{p.id}}</span><span class="${{isR ? 'text-emerald-500 font-bold' : 'text-gray-400'}}">${{isR ? '✓' : '○'}}</span>`;
        list.appendChild(b);
      }});
      document.getElementById('progress-text').innerText = `${{rated}} / ${{plans.length}} Rated`;
      document.getElementById('sidebar-rated').innerText = `${{rated}}/20`;
    }}

    function loadPlan(idx) {{
      currentIndex = idx;
      const p = plans[idx];
      document.getElementById('current-pid').innerText = p.id;
      document.getElementById('current-fname').innerText = p.filename;

      const tag = document.getElementById('badge-routed-tag');
      if (p.fallback_triggered) {{
        tag.className = "text-[10px] px-2 py-0.5 rounded font-bold bg-amber-500/10 text-amber-600 border border-amber-500/20";
        tag.innerText = "FALLBACK TRIGGERED";
      }} else {{
        tag.className = "text-[10px] px-2 py-0.5 rounded font-bold bg-emerald-500/10 text-emerald-600 border border-emerald-500/20";
        tag.innerText = "CHAMPION PATH";
      }}

      document.getElementById('img-orig').src = p.orig_b64;
      document.getElementById('img-champ').src = p.champ_b64;
      document.getElementById('img-routed').src = p.routed_b64;

      document.getElementById('champ-stats').innerText = `${{p.champ_rooms}} rms | ${{p.champ_area}} m2 | ${{p.champ_valid ? 'Valid' : 'Invalid'}}`;
      document.getElementById('routed-stats').innerText = `${{p.routed_rooms}} rms | ${{p.routed_area}} m2 | ${{p.routed_valid ? 'Valid' : 'Invalid'}}`;

      const r = ratings[p.id] || {{}};
      highlight('champ', r.champ);
      highlight('routed', r.routed);
      highlightPref(r.pref);
      renderSidebar();
    }}

    function navPlan(d) {{
      const ni = currentIndex + d;
      if (ni >= 0 && ni < plans.length) loadPlan(ni);
    }}

    function setScore(model, score) {{
      const pid = plans[currentIndex].id;
      if (!ratings[pid]) ratings[pid] = {{}};
      ratings[pid][model] = score;
      localStorage.setItem('archigen_champ_vs_routed_ratings', JSON.stringify(ratings));
      highlight(model, score);
      renderSidebar();
      updateStats();
    }}

    function highlight(model, score) {{
      ['usable', 'needs', 'unusable'].forEach(k => {{
        const b = document.getElementById(`btn-${{model}}-${{k}}`);
        if (b) b.classList.remove('bg-emerald-500', 'bg-amber-500', 'bg-rose-500', 'text-white');
      }});
      if (score === 'usable') document.getElementById(`btn-${{model}}-usable`).classList.add('bg-emerald-500', 'text-white');
      if (score === 'needs_fixes') document.getElementById(`btn-${{model}}-needs`).classList.add('bg-amber-500', 'text-white');
      if (score === 'unusable') document.getElementById(`btn-${{model}}-unusable`).classList.add('bg-rose-500', 'text-white');
    }}

    function setPref(pref) {{
      const pid = plans[currentIndex].id;
      if (!ratings[pid]) ratings[pid] = {{}};
      ratings[pid].pref = pref;
      localStorage.setItem('archigen_champ_vs_routed_ratings', JSON.stringify(ratings));
      highlightPref(pref);
      updateStats();
    }}

    function highlightPref(pref) {{
      ['champ', 'tie', 'routed'].forEach(k => {{
        document.getElementById(`btn-pref-${{k}}`).classList.remove('bg-blue-600', 'text-white');
      }});
      if (pref) document.getElementById(`btn-pref-${{pref}}`).classList.add('bg-blue-600', 'text-white');
    }}

    function updateStats() {{
      let cU = 0, cTot = 0, rU = 0, rTot = 0, rWin = 0, cWin = 0;
      plans.forEach(p => {{
        const r = ratings[p.id];
        if (!r) return;
        if (r.champ) {{ cTot++; if (r.champ === 'usable') cU++; }}
        if (r.routed) {{ rTot++; if (r.routed === 'usable') rU++; }}
        if (r.pref === 'routed') rWin++;
        if (r.pref === 'champ') cWin++;
      }});
      if (cTot > 0) document.getElementById('stat-champ').innerText = `${{cU}} / ${{cTot}} (${{Math.round(cU/cTot*100)}}%)`;
      if (rTot > 0) document.getElementById('stat-routed').innerText = `${{rU}} / ${{rTot}} (${{Math.round(rU/rTot*100)}}%)`;
      if (rWin + cWin > 0) document.getElementById('stat-win').innerText = `${{Math.round(rWin/(rWin+cWin)*100)}}% Win`;
    }}

    function exportRatings() {{
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ratings, null, 2));
      const a = document.createElement('a');
      a.href = dataStr;
      a.download = "dev_champion_vs_routed_human_ratings.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
    }}

    window.onload = init;
  </script>
</body>
</html>
"""
    html_p = ARTIFACT_DIR / "dev_champion_vs_routed_rater.html"
    with open(html_p, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"[DEV EVAL] Dedicated Champion vs Routed tool written to {html_p}")

if __name__ == "__main__":
    main()
