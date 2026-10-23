"""
Interactive Blind Rating Tool & Montage Generator for Dev Set (N=20).
Blinds Champion v1.1 vs Challenger with fixed seed=42 pseudorandom assignment.
Generates:
1. High-resolution comparison montage grid (dev_blind_eval_montage.png).
2. Interactive HTML side-by-side rating tool (dev_blind_rater.html) with Tailwind UI.
3. Blind mapping key (dev_blind_key.json).
"""

import sys
import json
import base64
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_DIR = Path("C:/Users/ayebk/.gemini/antigravity/brain/387c7b87-863d-4c00-831b-93abdc64d4d4")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def encode_img_base64(img_path: Path, max_size=512) -> str:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
        import io
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    manifest_p = PROJECT_ROOT / "data" / "dev_external_plans" / "dev_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    champ_dir = PROJECT_ROOT / "outputs" / "dev_results" / "baseline_v1_1"
    chal_dir = PROJECT_ROOT / "outputs" / "dev_results" / "restyled_challenger"
    blend_dir = PROJECT_ROOT / "outputs" / "dev_results" / "blend_alpha_0_3"

    random.seed(42)
    blind_key = {}
    plan_data = []

    # 1. Prepare blinded metadata
    for idx, item in enumerate(manifest):
        pid = item["id"]
        fname = item["filename"]
        orig_img_path = manifest_p.parent / fname

        champ_img_path = champ_dir / f"{pid}_overlay.png"
        chal_img_path = chal_dir / f"{pid}_overlay.png"
        blend_img_path = blend_dir / f"{pid}_overlay.png"

        # Blind flip: 0 = A:Champ, B:Chal; 1 = A:Chal, B:Champ
        flip = random.choice([0, 1])
        if flip == 0:
            model_a_src = "Champion v1.1"
            model_b_src = "Challenger"
            img_a_path = champ_img_path
            img_b_path = chal_img_path
        else:
            model_a_src = "Challenger"
            model_b_src = "Champion v1.1"
            img_a_path = chal_img_path
            img_b_path = champ_img_path

        blind_key[pid] = {
            "Model A": model_a_src,
            "Model B": model_b_src,
            "flip": flip
        }

        plan_data.append({
            "id": pid,
            "title": item.get("title", pid),
            "filename": fname,
            "orig_b64": encode_img_base64(orig_img_path, max_size=400),
            "model_a_b64": encode_img_base64(img_a_path, max_size=400),
            "model_b_b64": encode_img_base64(img_b_path, max_size=400),
            "blend_b64": encode_img_base64(blend_img_path, max_size=400) if blend_img_path.exists() else None,
            "model_a_src": model_a_src,
            "model_b_src": model_b_src
        })

    # Save blind key
    key_path = PROJECT_ROOT / "outputs" / "dev_blind_key.json"
    with open(key_path, "w") as f:
        json.dump(blind_key, f, indent=2)
    print(f"[BLIND EVAL] Saved blind key with seed=42 to {key_path}")

    # 2. Build High-Resolution Montage Grid (5 rows x 4 plans or 20 rows of 3 columns)
    # Let's generate a clean 20-row montage: Col 1: Original, Col 2: Model A, Col 3: Model B
    row_h = 240
    col_w = 240
    montage_w = col_w * 3 + 120
    montage_h = row_h * 20 + 80

    montage = Image.new("RGB", (montage_w, montage_h), (245, 247, 250))
    draw = ImageDraw.Draw(montage)

    # Header
    draw.rectangle([(0, 0), (montage_w, 70)], fill=(30, 41, 59))
    draw.text((20, 15), "ARCHIGEN 3D - BLIND HUMAN RATING STUDY (N=20 DEV PLANS)", fill=(255, 255, 255))
    draw.text((20, 42), "Columns: [Plan ID] | Original Blueprint | Blind Model A | Blind Model B", fill=(203, 213, 225))

    for idx, p in enumerate(plan_data):
        y_top = 80 + idx * row_h
        # Background stripe
        stripe_col = (255, 255, 255) if idx % 2 == 0 else (241, 245, 249)
        draw.rectangle([(0, y_top), (montage_w, y_top + row_h - 2)], fill=stripe_col)

        # Plan Label
        draw.text((15, y_top + 100), f"{p['id']}", fill=(15, 23, 42))

        # Original
        orig_img = Image.open(manifest_p.parent / p["filename"]).convert("RGB")
        orig_img.thumbnail((col_w - 10, row_h - 20), Image.Resampling.BILINEAR)
        montage.paste(orig_img, (80, y_top + 10))

        # Model A
        a_path = (champ_dir if p["model_a_src"] == "Champion v1.1" else chal_dir) / f"{p['id']}_overlay.png"
        im_a = Image.open(a_path).convert("RGB")
        im_a.thumbnail((col_w - 10, row_h - 20), Image.Resampling.BILINEAR)
        montage.paste(im_a, (80 + col_w, y_top + 10))

        # Model B
        b_path = (champ_dir if p["model_b_src"] == "Champion v1.1" else chal_dir) / f"{p['id']}_overlay.png"
        im_b = Image.open(b_path).convert("RGB")
        im_b.thumbnail((col_w - 10, row_h - 20), Image.Resampling.BILINEAR)
        montage.paste(im_b, (80 + col_w * 2, y_top + 10))

    montage_path = ARTIFACT_DIR / "dev_blind_eval_montage.png"
    montage.save(montage_path, quality=85)
    print(f"[BLIND EVAL] Saved montage grid to {montage_path}")

    # 3. Generate Interactive HTML Rating Application
    html_content = generate_html_tool(plan_data, blind_key)
    html_path = ARTIFACT_DIR / "dev_blind_rater.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[BLIND EVAL] Saved interactive HTML tool to {html_path}")

def generate_html_tool(plan_data, blind_key):
    data_json = json.dumps(plan_data)
    key_json = json.dumps(blind_key)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ArchiGen 3D - Blind Human Rating Study</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .active-nav {{ border-color: #3b82f6; background-color: rgba(59, 130, 246, 0.1); color: #3b82f6; font-weight: 600; }}
    .badge-usable {{ background-color: #dcfce7; color: #166534; border: 1px solid #86efac; }}
    .badge-needs {{ background-color: #fef9c3; color: #854d0e; border: 1px solid #fde047; }}
    .badge-unusable {{ background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased min-h-screen p-4 md:p-6 font-sans">

  <!-- Header & Telemetry Bar -->
  <header class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 mb-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
    <div>
      <div class="flex items-center gap-3">
        <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider rounded-md bg-blue-500/10 text-blue-500 border border-blue-500/20">Human In The Loop</span>
        <h1 class="text-xl md:text-2xl font-bold tracking-tight">Blind Dev Set Usability Benchmark (N=20)</h1>
      </div>
      <p class="text-[var(--muted-foreground)] text-sm mt-1">
        Side-by-side evaluation of 20 out-of-domain CAD/raster plans. Model identities (Champion v1.1 vs Challenger) are randomized per plan.
      </p>
    </div>

    <!-- Live Progress & Actions -->
    <div class="flex items-center gap-3">
      <div class="text-right">
        <div class="text-xs uppercase tracking-wider text-[var(--muted-foreground)]">Rated Progress</div>
        <div class="text-base font-bold text-blue-500" id="progress-text">0 / 20 Plans</div>
      </div>
      <button onclick="revealBlindKey()" id="btn-reveal" class="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-xl shadow transition flex items-center gap-2">
        <span>🔓 Reveal Blind Key</span>
      </button>
      <button onclick="exportRatings()" class="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-xl shadow transition flex items-center gap-2">
        <span>💾 Export Ratings JSON</span>
      </button>
    </div>
  </header>

  <!-- Main Grid Layout -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left Sidebar: Plan Selector -->
    <div class="lg:col-span-3 bg-[var(--card)] border border-[var(--border)] rounded-2xl p-4 shadow-sm h-[780px] flex flex-col">
      <div class="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3 px-2 flex justify-between">
        <span>Plans (20)</span>
        <span id="sidebar-rated-count">0 rated</span>
      </div>
      <div class="overflow-y-auto space-y-1.5 flex-1 pr-1" id="plan-list">
        <!-- Rendered dynamically -->
      </div>
    </div>

    <!-- Center/Right: Current Plan Inspection & Rating Form -->
    <div class="lg:col-span-9 space-y-6">

      <!-- Plan Info Card -->
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-5 shadow-sm">
        <div class="flex items-center justify-between mb-4">
          <div>
            <span class="text-xs font-mono font-bold text-blue-500 px-2 py-0.5 rounded bg-blue-500/10" id="current-pid">dev_01</span>
            <span class="text-sm font-medium text-[var(--foreground)] ml-2" id="current-title"></span>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="navPlan(-1)" class="px-3 py-1.5 text-xs font-medium border border-[var(--border)] rounded-lg hover:bg-[var(--accent)] transition">‹ Prev</button>
            <button onclick="navPlan(1)" class="px-3 py-1.5 text-xs font-medium border border-[var(--border)] rounded-lg hover:bg-[var(--accent)] transition">Next ›</button>
          </div>
        </div>

        <!-- 3-Column Visual Comparison -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <!-- Col 1: Original Floorplan -->
          <div class="border border-[var(--border)] rounded-xl overflow-hidden bg-black/5 flex flex-col">
            <div class="px-3 py-2 bg-[var(--accent)] border-b border-[var(--border)] text-xs font-semibold flex items-center justify-between">
              <span>Original Blueprint</span>
              <span class="text-[10px] text-[var(--muted-foreground)]">CAD / Raster</span>
            </div>
            <div class="p-2 flex-1 flex items-center justify-center bg-white min-h-[280px]">
              <img id="img-orig" src="" class="max-h-[280px] max-w-full object-contain cursor-zoom-in" onclick="zoomImage(this.src)" />
            </div>
          </div>

          <!-- Col 2: Model A -->
          <div class="border border-[var(--border)] rounded-xl overflow-hidden bg-black/5 flex flex-col">
            <div class="px-3 py-2 bg-[var(--accent)] border-b border-[var(--border)] text-xs font-semibold flex items-center justify-between">
              <span class="flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-blue-500"></span> Model A
              </span>
              <span id="revealed-model-a" class="hidden px-1.5 py-0.5 text-[10px] font-bold rounded bg-indigo-500/10 text-indigo-500"></span>
            </div>
            <div class="p-2 flex-1 flex items-center justify-center bg-white min-h-[280px]">
              <img id="img-a" src="" class="max-h-[280px] max-w-full object-contain cursor-zoom-in" onclick="zoomImage(this.src)" />
            </div>
          </div>

          <!-- Col 3: Model B -->
          <div class="border border-[var(--border)] rounded-xl overflow-hidden bg-black/5 flex flex-col">
            <div class="px-3 py-2 bg-[var(--accent)] border-b border-[var(--border)] text-xs font-semibold flex items-center justify-between">
              <span class="flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-purple-500"></span> Model B
              </span>
              <span id="revealed-model-b" class="hidden px-1.5 py-0.5 text-[10px] font-bold rounded bg-indigo-500/10 text-indigo-500"></span>
            </div>
            <div class="p-2 flex-1 flex items-center justify-center bg-white min-h-[280px]">
              <img id="img-b" src="" class="max-h-[280px] max-w-full object-contain cursor-zoom-in" onclick="zoomImage(this.src)" />
            </div>
          </div>
        </div>

        <!-- Rating Controls (Side-by-Side Dual Form) -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-[var(--border)]">

          <!-- Rating Form Model A -->
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--card)] space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="text-sm font-bold flex items-center gap-2">
                <span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span> Rate Model A
              </h3>
              <span id="badge-a" class="text-xs font-semibold px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">Unrated</span>
            </div>

            <div>
              <label class="block text-xs font-medium text-[var(--muted-foreground)] mb-1.5">Usability Level</label>
              <div class="grid grid-cols-3 gap-2">
                <button onclick="setScore('A', 'usable')" id="btn-a-usable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-emerald-500/10 hover:text-emerald-600 transition">
                  ✓ Usable (2)
                </button>
                <button onclick="setScore('A', 'needs_fixes')" id="btn-a-needs" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-amber-500/10 hover:text-amber-600 transition">
                  ⚡ Needs Fixes (1)
                </button>
                <button onclick="setScore('A', 'unusable')" id="btn-a-unusable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-rose-500/10 hover:text-rose-600 transition">
                  ✕ Unusable (0)
                </button>
              </div>
            </div>

            <div>
              <label class="block text-xs font-medium text-[var(--muted-foreground)] mb-1">Room Partitioning</label>
              <select id="sel-a-room" onchange="saveCurrentPlanState()" class="w-full text-xs bg-[var(--card)] border border-[var(--border)] rounded-lg p-2 text-[var(--foreground)]">
                <option value="realistic">Realistic division (correct rooms)</option>
                <option value="over_split">Over-split (fragmented wall slivers)</option>
                <option value="under_split">Under-split (missing rooms)</option>
                <option value="collapsed">Zero rooms detected</option>
              </select>
            </div>
          </div>

          <!-- Rating Form Model B -->
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--card)] space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="text-sm font-bold flex items-center gap-2">
                <span class="w-2.5 h-2.5 rounded-full bg-purple-500"></span> Rate Model B
              </h3>
              <span id="badge-b" class="text-xs font-semibold px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">Unrated</span>
            </div>

            <div>
              <label class="block text-xs font-medium text-[var(--muted-foreground)] mb-1.5">Usability Level</label>
              <div class="grid grid-cols-3 gap-2">
                <button onclick="setScore('B', 'usable')" id="btn-b-usable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-emerald-500/10 hover:text-emerald-600 transition">
                  ✓ Usable (2)
                </button>
                <button onclick="setScore('B', 'needs_fixes')" id="btn-b-needs" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-amber-500/10 hover:text-amber-600 transition">
                  ⚡ Needs Fixes (1)
                </button>
                <button onclick="setScore('B', 'unusable')" id="btn-b-unusable" class="px-2.5 py-2 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-rose-500/10 hover:text-rose-600 transition">
                  ✕ Unusable (0)
                </button>
              </div>
            </div>

            <div>
              <label class="block text-xs font-medium text-[var(--muted-foreground)] mb-1">Room Partitioning</label>
              <select id="sel-b-room" onchange="saveCurrentPlanState()" class="w-full text-xs bg-[var(--card)] border border-[var(--border)] rounded-lg p-2 text-[var(--foreground)]">
                <option value="realistic">Realistic division (correct rooms)</option>
                <option value="over_split">Over-split (fragmented wall slivers)</option>
                <option value="under_split">Under-split (missing rooms)</option>
                <option value="collapsed">Zero rooms detected</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Comparative Preference -->
        <div class="mt-4 pt-4 border-t border-[var(--border)] flex flex-wrap items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <span class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)]">Head-to-Head Preference:</span>
            <div class="flex items-center gap-2">
              <button onclick="setPreference('A')" id="btn-pref-a" class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-blue-500/10 hover:text-blue-600 transition">
                Prefer Model A
              </button>
              <button onclick="setPreference('TIE')" id="btn-pref-tie" class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-gray-500/10 transition">
                Tie / Equal
              </button>
              <button onclick="setPreference('B')" id="btn-pref-b" class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-[var(--border)] hover:bg-purple-500/10 hover:text-purple-600 transition">
                Prefer Model B
              </button>
            </div>
          </div>

          <div class="flex items-center gap-2">
            <button onclick="saveAndNext()" class="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-xl shadow transition">
              Save & Next Plan →
            </button>
          </div>
        </div>
      </div>

      <!-- Live Summary & Blind Decoder Section -->
      <div id="results-card" class="bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-sm">
        <h2 class="text-lg font-bold mb-3 flex items-center justify-between">
          <span>Human Usability Scorecard</span>
          <span id="revealed-tag" class="hidden text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 border border-emerald-500/20">DECODED</span>
        </h2>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Champion v1.1 Usable</div>
            <div class="text-2xl font-bold mt-1" id="stat-champ-usable">—</div>
            <div class="text-[11px] text-[var(--muted-foreground)]" id="stat-champ-sub">Awaiting ratings</div>
          </div>
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Challenger Usable</div>
            <div class="text-2xl font-bold mt-1 text-purple-600" id="stat-chal-usable">—</div>
            <div class="text-[11px] text-[var(--muted-foreground)]" id="stat-chal-sub">Awaiting ratings</div>
          </div>
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Challenger Win Rate</div>
            <div class="text-2xl font-bold mt-1 text-emerald-600" id="stat-win-rate">—</div>
            <div class="text-[11px] text-[var(--muted-foreground)]" id="stat-win-sub">Head-to-head</div>
          </div>
          <div class="p-4 rounded-xl border border-[var(--border)] bg-[var(--accent)]">
            <div class="text-xs text-[var(--muted-foreground)] uppercase">Over-Split Rate</div>
            <div class="text-2xl font-bold mt-1 text-rose-500" id="stat-oversplit">—</div>
            <div class="text-[11px] text-[var(--muted-foreground)]" id="stat-oversplit-sub">Human flagged</div>
          </div>
        </div>

        <div class="text-xs text-[var(--muted-foreground)] leading-relaxed">
          <strong class="text-[var(--foreground)]">Evaluation Standard:</strong> Usable score reflects true layout viability (watertight exterior walls, plausible room boundaries without severe fragmentation). This human metric formally replaces the unvalidated heuristic rule (shell valid + 3 rooms + 25m²).
        </div>
      </div>

    </div>
  </div>

  <script>
    const plans = {data_json};
    const blindKey = {key_json};

    let currentIndex = 0;
    let isRevealed = false;

    // Load existing ratings from localStorage or init empty
    let ratings = JSON.parse(localStorage.getItem('archigen_dev_blind_ratings') || '{{}}');

    function init() {{
      renderSidebar();
      loadPlan(currentIndex);
      updateSummary();
    }}

    function renderSidebar() {{
      const listEl = document.getElementById('plan-list');
      listEl.innerHTML = '';
      let ratedCount = 0;

      plans.forEach((p, idx) => {{
        const r = ratings[p.id];
        const isRated = r && r.score_a && r.score_b;
        if (isRated) ratedCount++;

        const item = document.createElement('button');
        item.className = `w-full text-left px-3 py-2.5 rounded-xl border text-xs flex items-center justify-between transition ${{idx === currentIndex ? 'active-nav' : 'border-[var(--border)] hover:bg-[var(--accent)]'}}`;
        item.onclick = () => loadPlan(idx);

        let statusBadge = '<span class="text-gray-400">○</span>';
        if (isRated) {{
          statusBadge = '<span class="text-emerald-500 font-bold">✓</span>';
        }}

        item.innerHTML = `
          <div class="flex items-center gap-2 truncate">
            <span class="font-mono font-bold">${{p.id}}</span>
            <span class="truncate text-[var(--muted-foreground)]">${{p.title}}</span>
          </div>
          <div>${{statusBadge}}</div>
        `;
        listEl.appendChild(item);
      }});

      document.getElementById('progress-text').innerText = `${{ratedCount}} / ${{plans.length}} Plans`;
      document.getElementById('sidebar-rated-count').innerText = `${{ratedCount}} rated`;
    }}

    function loadPlan(idx) {{
      currentIndex = idx;
      const p = plans[idx];

      document.getElementById('current-pid').innerText = p.id;
      document.getElementById('current-title').innerText = p.title;

      document.getElementById('img-orig').src = p.orig_b64;
      document.getElementById('img-a').src = p.model_a_b64;
      document.getElementById('img-b').src = p.model_b_b64;

      if (isRevealed) {{
        document.getElementById('revealed-model-a').innerText = p.model_a_src;
        document.getElementById('revealed-model-b').innerText = p.model_b_src;
        document.getElementById('revealed-model-a').classList.remove('hidden');
        document.getElementById('revealed-model-b').classList.remove('hidden');
      }}

      // Restore form state
      const r = ratings[p.id] || {{}};
      updateScoreButtons('A', r.score_a);
      updateScoreButtons('B', r.score_b);
      updatePrefButtons(r.preference);

      document.getElementById('sel-a-room').value = r.room_a || 'realistic';
      document.getElementById('sel-b-room').value = r.room_b || 'realistic';

      renderSidebar();
    }}

    function navPlan(delta) {{
      const newIdx = currentIndex + delta;
      if (newIdx >= 0 && newIdx < plans.length) {{
        loadPlan(newIdx);
      }}
    }}

    function setScore(model, score) {{
      const pid = plans[currentIndex].id;
      if (!ratings[pid]) ratings[pid] = {{}};
      ratings[pid]['score_' + model.toLowerCase()] = score;
      ratings[pid]['room_' + model.toLowerCase()] = document.getElementById('sel-' + model.toLowerCase() + '-room').value;
      localStorage.setItem('archigen_dev_blind_ratings', JSON.stringify(ratings));

      updateScoreButtons(model, score);
      renderSidebar();
      updateSummary();
    }}

    function updateScoreButtons(model, score) {{
      const prefix = 'btn-' + model.toLowerCase() + '-';
      ['usable', 'needs', 'unusable'].forEach(s => {{
        const btn = document.getElementById(prefix + s);
        if (btn) {{
          btn.classList.remove('bg-emerald-500', 'bg-amber-500', 'bg-rose-500', 'text-white', 'border-transparent');
          btn.classList.add('border-[var(--border)]');
        }}
      }});

      const badge = document.getElementById('badge-' + model.toLowerCase());
      if (score === 'usable') {{
        const b = document.getElementById(prefix + 'usable');
        b.classList.add('bg-emerald-500', 'text-white', 'border-transparent');
        badge.className = 'text-xs font-semibold px-2 py-0.5 rounded-full badge-usable';
        badge.innerText = 'Usable (2)';
      }} else if (score === 'needs_fixes') {{
        const b = document.getElementById(prefix + 'needs');
        b.classList.add('bg-amber-500', 'text-white', 'border-transparent');
        badge.className = 'text-xs font-semibold px-2 py-0.5 rounded-full badge-needs';
        badge.innerText = 'Needs Fixes (1)';
      }} else if (score === 'unusable') {{
        const b = document.getElementById(prefix + 'unusable');
        b.classList.add('bg-rose-500', 'text-white', 'border-transparent');
        badge.className = 'text-xs font-semibold px-2 py-0.5 rounded-full badge-unusable';
        badge.innerText = 'Unusable (0)';
      }} else {{
        badge.className = 'text-xs font-semibold px-2 py-0.5 rounded-full bg-gray-100 text-gray-600';
        badge.innerText = 'Unrated';
      }}
    }}

    function setPreference(pref) {{
      const pid = plans[currentIndex].id;
      if (!ratings[pid]) ratings[pid] = {{}};
      ratings[pid].preference = pref;
      localStorage.setItem('archigen_dev_blind_ratings', JSON.stringify(ratings));

      updatePrefButtons(pref);
      updateSummary();
    }}

    function updatePrefButtons(pref) {{
      ['a', 'tie', 'b'].forEach(k => {{
        const btn = document.getElementById('btn-pref-' + k);
        btn.classList.remove('bg-blue-600', 'bg-purple-600', 'bg-gray-600', 'text-white');
      }});

      if (pref === 'A') {{
        document.getElementById('btn-pref-a').classList.add('bg-blue-600', 'text-white');
      }} else if (pref === 'B') {{
        document.getElementById('btn-pref-b').classList.add('bg-purple-600', 'text-white');
      }} else if (pref === 'TIE') {{
        document.getElementById('btn-pref-tie').classList.add('bg-gray-600', 'text-white');
      }}
    }}

    function saveCurrentPlanState() {{
      const pid = plans[currentIndex].id;
      if (!ratings[pid]) ratings[pid] = {{}};
      ratings[pid].room_a = document.getElementById('sel-a-room').value;
      ratings[pid].room_b = document.getElementById('sel-b-room').value;
      localStorage.setItem('archigen_dev_blind_ratings', JSON.stringify(ratings));
      updateSummary();
    }}

    function saveAndNext() {{
      saveCurrentPlanState();
      if (currentIndex < plans.length - 1) {{
        loadPlan(currentIndex + 1);
      }}
    }}

    function revealBlindKey() {{
      isRevealed = true;
      document.getElementById('revealed-tag').classList.remove('hidden');
      document.getElementById('btn-reveal').innerHTML = '<span>🔓 Blind Key Active</span>';
      loadPlan(currentIndex);
      updateSummary();
    }}

    function updateSummary() {{
      let champUsable = 0, champTotal = 0;
      let chalUsable = 0, chalTotal = 0;
      let chalWins = 0, champWins = 0, ties = 0;
      let chalOverSplit = 0;

      plans.forEach(p => {{
        const r = ratings[p.id];
        if (!r) return;

        const aIsChamp = (p.model_a_src === 'Champion v1.1');
        const champScore = aIsChamp ? r.score_a : r.score_b;
        const chalScore = aIsChamp ? r.score_b : r.score_a;
        const chalRoom = aIsChamp ? r.room_b : r.room_a;

        if (champScore) {{
          champTotal++;
          if (champScore === 'usable') champUsable++;
        }}
        if (chalScore) {{
          chalTotal++;
          if (chalScore === 'usable') chalUsable++;
        }}
        if (chalRoom === 'over_split') {{
          chalOverSplit++;
        }}

        if (r.preference) {{
          if (r.preference === 'TIE') ties++;
          else if ((r.preference === 'A' && aIsChamp) || (r.preference === 'B' && !aIsChamp)) champWins++;
          else chalWins++;
        }}
      }});

      if (champTotal > 0) {{
        document.getElementById('stat-champ-usable').innerText = `${{champUsable}} / ${{champTotal}} (${{Math.round(champUsable/champTotal*100)}}%)`;
        document.getElementById('stat-champ-sub').innerText = `${{champTotal}} plans evaluated`;
      }}
      if (chalTotal > 0) {{
        document.getElementById('stat-chal-usable').innerText = `${{chalUsable}} / ${{chalTotal}} (${{Math.round(chalUsable/chalTotal*100)}}%)`;
        document.getElementById('stat-chal-sub').innerText = `${{chalTotal}} plans evaluated`;
      }}
      const totPref = chalWins + champWins + ties;
      if (totPref > 0) {{
        document.getElementById('stat-win-rate').innerText = `${{Math.round(chalWins/totPref*100)}}% Win`;
        document.getElementById('stat-win-sub').innerText = `${{chalWins}}W / ${{champWins}}L / ${{ties}}T`;
      }}
      if (chalTotal > 0) {{
        document.getElementById('stat-oversplit').innerText = `${{chalOverSplit}} / ${{chalTotal}} (${{Math.round(chalOverSplit/chalTotal*100)}}%)`;
      }}
    }}

    function exportRatings() {{
      const exportObj = {{
        timestamp: new Date().toISOString(),
        total_plans: plans.length,
        blind_key: blindKey,
        ratings: ratings
      }};
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", "dev_human_ratings.json");
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    }}

    function zoomImage(src) {{
      const modal = document.createElement('div');
      modal.className = 'fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4 cursor-zoom-out';
      modal.onclick = () => modal.remove();
      modal.innerHTML = `<img src="${{src}}" class="max-w-full max-h-full rounded-lg shadow-2xl object-contain" />`;
      document.body.appendChild(modal);
    }}

    window.onload = init;
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
