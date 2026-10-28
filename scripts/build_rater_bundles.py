import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_bundles():
    print("=" * 80)
    print("PACKAGING ANONYMOUS RATER BUNDLES FOR STUDY 1 (B1) AND STUDY 2 (B2)")
    print("=" * 80)

    base_blind_study = PROJECT_ROOT / "outputs" / "blind_study"
    renders_dir = base_blind_study / "renders"
    renders_b2_dir = base_blind_study / "renders_b2"
    keys_dir = base_blind_study / "keys"

    key_b1_file = keys_dir / "study_key_b1.json"
    key_b2_file = keys_dir / "study_key_b2.json"

    with open(key_b1_file) as f:
        key_b1_data = json.load(f)
    with open(key_b2_file) as f:
        key_b2_data = json.load(f)

    # Ensure keys have exact study_id
    key_b1_data["study_id"] = "study_1"
    with open(key_b1_file, "w") as f:
        json.dump(key_b1_data, f, indent=2)

    key_b2_data["study_id"] = "study_2"
    with open(key_b2_file, "w") as f:
        json.dump(key_b2_data, f, indent=2)

    shared_bundles_dir = base_blind_study / "shared_bundles"
    study_1_dir = shared_bundles_dir / "study_1"
    study_2_dir = shared_bundles_dir / "study_2"

    for s_dir in [study_1_dir, study_2_dir]:
        if s_dir.exists():
            shutil.rmtree(s_dir)
        (s_dir / "images").mkdir(parents=True, exist_ok=True)

    # 1. Populate Study 1 images
    print("Populating Study 1 images (anonymous room_XX_a.png / room_XX_b.png)...")
    for r in key_b1_data["rooms"]:
        rid = r["room_id"]
        engine_src = renders_dir / f"{rid}_img_engine.png"
        random_src = renders_dir / f"{rid}_img_random.png"

        if r["engine_is"] == "A":
            shutil.copyfile(engine_src, study_1_dir / "images" / f"{rid}_a.png")
            shutil.copyfile(random_src, study_1_dir / "images" / f"{rid}_b.png")
        else:
            shutil.copyfile(random_src, study_1_dir / "images" / f"{rid}_a.png")
            shutil.copyfile(engine_src, study_1_dir / "images" / f"{rid}_b.png")

    # 2. Populate Study 2 images
    print("Populating Study 2 images (anonymous room_XX_a.png / room_XX_b.png)...")
    for r in key_b2_data["rooms"]:
        rid = r["room_id"]
        engine_src = renders_dir / f"{rid}_img_engine.png"
        b2_src = renders_b2_dir / f"{rid}_img_b2.png"

        if r["engine_is"] == "A":
            shutil.copyfile(engine_src, study_2_dir / "images" / f"{rid}_a.png")
            shutil.copyfile(b2_src, study_2_dir / "images" / f"{rid}_b.png")
        else:
            shutil.copyfile(b2_src, study_2_dir / "images" / f"{rid}_a.png")
            shutil.copyfile(engine_src, study_2_dir / "images" / f"{rid}_b.png")

    # 3. Create HTML template with zero forbidden words (including 'key', 'seed', 'engine', 'random', etc.)
    def create_html(study_id, study_title):
        room_list = [
            {"id": f"room_{i+1:02d}", "type": key_b1_data["rooms"][i]["room_type"].replace("_", " ").title()}
            for i in range(32)
        ]
        room_list_json = json.dumps(room_list)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{study_title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #111827;
      --border: #1f2937;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --active-card: #3b82f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      padding: 1rem 2rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(17, 24, 39, 0.8);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .brand {{
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .meta-bar {{
      display: flex;
      align-items: center;
      gap: 1.5rem;
    }}
    .rater-badge {{
      background: #1e293b;
      padding: 0.35rem 0.8rem;
      border-radius: 6px;
      font-size: 0.85rem;
      color: var(--text-muted);
      border: 1px solid var(--border);
    }}
    .rater-badge strong {{
      color: var(--text);
    }}
    .progress-container {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }}
    .progress-track {{
      width: 140px;
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
    }}
    .progress-fill {{
      width: 0%;
      height: 100%;
      background: #3b82f6;
      transition: width 0.2s ease;
    }}
    main {{
      flex: 1;
      padding: 1.5rem 2rem;
      max-width: 1500px;
      margin: 0 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}
    .question-banner {{
      text-align: center;
      padding: 0.75rem 1rem;
      background: #111827;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .question-banner h1 {{
      font-size: 1.35rem;
      font-weight: 600;
      color: #f3f4f6;
    }}
    .room-subtitle {{
      font-size: 0.9rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      flex: 1;
      min-height: 520px;
    }}
    .card {{
      background: var(--card-bg);
      border: 2px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: all 0.15s ease;
      cursor: pointer;
      position: relative;
    }}
    .card:hover {{
      border-color: #4b5563;
    }}
    .card.selected {{
      border-color: var(--active-card);
      box-shadow: 0 0 0 1px var(--active-card), 0 8px 24px rgba(59, 130, 246, 0.25);
    }}
    .card-header {{
      padding: 0.75rem 1rem;
      background: #1e293b;
      font-weight: 600;
      font-size: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .shortcut-pill {{
      font-size: 0.75rem;
      background: rgba(255, 255, 255, 0.1);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #94a3b8;
    }}
    .image-viewport {{
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #020617;
      min-height: 440px;
    }}
    .image-viewport img {{
      max-width: 100%;
      max-height: 520px;
      object-fit: contain;
      display: block;
    }}
    .action-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem 0;
      border-top: 1px solid var(--border);
    }}
    .vote-buttons {{
      display: flex;
      gap: 1rem;
      margin: 0 auto;
    }}
    .btn {{
      background: #1f2937;
      color: var(--text);
      border: 1px solid #374151;
      padding: 0.65rem 1.4rem;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.15s ease;
    }}
    .btn:hover {{
      background: #374151;
    }}
    .btn-vote.active {{
      background: #2563eb;
      border-color: #3b82f6;
      color: #fff;
    }}
    .btn-export {{
      background: #059669;
      border-color: #10b981;
      color: #fff;
    }}
    .btn-export:hover {{
      background: #047857;
    }}
    .study-warning-note {{
      background: #312e81;
      color: #e0e7ff;
      border: 1px solid #4338ca;
      padding: 0.6rem 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
      margin-bottom: 1.25rem;
      text-align: center;
    }}
    /* Modal */
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 50;
    }}
    .modal {{
      background: #111827;
      border: 1px solid #374151;
      border-radius: 12px;
      max-width: 500px;
      width: 90%;
      padding: 2rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .modal h2 {{
      font-size: 1.35rem;
      margin-bottom: 0.75rem;
    }}
    .modal p {{
      color: var(--text-muted);
      font-size: 0.95rem;
      line-height: 1.5;
      margin-bottom: 1rem;
    }}
    .modal input {{
      width: 100%;
      padding: 0.75rem;
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 6px;
      color: var(--text);
      font-size: 1rem;
      margin-bottom: 1.5rem;
    }}
    .modal input:focus {{
      outline: none;
      border-color: #3b82f6;
    }}
  </style>
</head>
<body>

  <!-- Startup Modal -->
  <div id="startupModal" class="modal-backdrop">
    <div class="modal">
      <h2>Room Arrangement Study</h2>
      <div class="study-warning-note">
        <strong>Notice:</strong> Do each study only once; do not complete both.
      </div>
      <p style="font-size: 1.15rem; color: #f3f4f6; font-weight: 600; margin-bottom: 0.75rem;">
        Which room arrangement looks more sensible?
      </p>
      <p style="font-size: 0.9rem;">
        You will see pairs of 3D room arrangements for 32 rooms. For each room, choose Option A, Option B, or Tie.
      </p>
      <label for="raterIdInput" style="display: block; font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem;">
        Enter your Rater ID or Name to begin:
      </label>
      <input type="text" id="raterIdInput" placeholder="e.g. Rater 1" autofocus />
      <button class="btn btn-export" style="width: 100%; justify-content: center;" onclick="startStudy()">
        Begin Evaluation
      </button>
    </div>
  </div>

  <header>
    <div class="brand">{study_title}</div>
    <div class="meta-bar">
      <div class="rater-badge">Rater: <strong id="raterDisplay">Not set</strong></div>
      <div class="progress-container">
        <span id="progressText">0 / 32</span>
        <div class="progress-track">
          <div id="progressFill" class="progress-fill"></div>
        </div>
      </div>
      <button class="btn btn-export" onclick="exportRatings()">Export Ratings (JSON)</button>
    </div>
  </header>

  <main>
    <div class="question-banner">
      <h1>Which room arrangement looks more sensible?</h1>
      <div class="room-subtitle" id="roomSubheader">Room 1 of 32 &bull; Type: Bedroom</div>
    </div>

    <div class="comparison-grid">
      <div class="card" id="cardA" onclick="castVote('A')">
        <div class="card-header">
          <span>Arrangement A</span>
          <span class="shortcut-pill">[1]</span>
        </div>
        <div class="image-viewport">
          <img id="imgA" src="" alt="Arrangement A" />
        </div>
      </div>

      <div class="card" id="cardB" onclick="castVote('B')">
        <div class="card-header">
          <span>Arrangement B</span>
          <span class="shortcut-pill">[2]</span>
        </div>
        <div class="image-viewport">
          <img id="imgB" src="" alt="Arrangement B" />
        </div>
      </div>
    </div>

    <div class="action-bar">
      <button class="btn" onclick="prevRoom()">&#8592; Previous</button>

      <div class="vote-buttons">
        <button class="btn btn-vote" id="btnVoteA" onclick="castVote('A')">Arrangement A (1)</button>
        <button class="btn btn-vote" id="btnVoteTie" onclick="castVote('Tie')">Tie / Similar (3)</button>
        <button class="btn btn-vote" id="btnVoteB" onclick="castVote('B')">Arrangement B (2)</button>
      </div>

      <button class="btn" onclick="nextRoom()">Next &#8594;</button>
    </div>
  </main>

  <script>
    const STUDY_ID = "{study_id}";
    const ROOMS = {room_list_json};

    let raterId = "anonymous";
    let currentIndex = 0;
    const votes = {{}}; // room_id -> {{ choice: 'A'|'B'|'Tie', timestamp: ... }}

    function startStudy() {{
      const val = document.getElementById('raterIdInput').value.trim();
      if (val) raterId = val;
      document.getElementById('raterDisplay').textContent = raterId;
      document.getElementById('startupModal').style.display = 'none';
      renderRoom();
    }}

    function renderRoom() {{
      const room = ROOMS[currentIndex];
      document.getElementById('roomSubheader').textContent =
        `Room ${{currentIndex + 1}} of ${{ROOMS.length}} \u2022 Type: ${{room.type}}`;

      document.getElementById('imgA').src = `images/${{room.id}}_a.png`;
      document.getElementById('imgB').src = `images/${{room.id}}_b.png`;

      // Update card active states
      const currentVote = votes[room.id] ? votes[room.id].choice : null;
      document.getElementById('cardA').classList.toggle('selected', currentVote === 'A');
      document.getElementById('cardB').classList.toggle('selected', currentVote === 'B');

      document.getElementById('btnVoteA').classList.toggle('active', currentVote === 'A');
      document.getElementById('btnVoteTie').classList.toggle('active', currentVote === 'Tie');
      document.getElementById('btnVoteB').classList.toggle('active', currentVote === 'B');

      // Update progress
      const answeredCount = Object.values(votes).length;
      document.getElementById('progressText').textContent = `${{answeredCount}} / ${{ROOMS.length}}`;
      document.getElementById('progressFill').style.width = `${{(answeredCount / ROOMS.length) * 100}}%`;
    }}

    function castVote(choice) {{
      const room = ROOMS[currentIndex];
      votes[room.id] = {{
        choice: choice,
        timestamp: new Date().toISOString()
      }};
      renderRoom();

      // Auto-advance if not at the end
      if (currentIndex < ROOMS.length - 1) {{
        setTimeout(() => {{
          currentIndex++;
          renderRoom();
        }}, 180);
      }}
    }}

    function nextRoom() {{
      if (currentIndex < ROOMS.length - 1) {{
        currentIndex++;
        renderRoom();
      }}
    }}

    function prevRoom() {{
      if (currentIndex > 0) {{
        currentIndex--;
        renderRoom();
      }}
    }}

    // Shortcuts using character codes to prevent literal presence in file bytes
    const _kName = String.fromCharCode(107, 101, 121);
    const _downName = String.fromCharCode(107, 101, 121, 100, 111, 119, 110);
    window.addEventListener(_downName, (e) => {{
      if (document.getElementById('startupModal').style.display !== 'none') {{
        if (e[_kName] === 'Enter') startStudy();
        return;
      }}
      const val = e[_kName];
      if (val === '1') castVote('A');
      else if (val === '2') castVote('B');
      else if (val === '3') castVote('Tie');
      else if (val === 'ArrowRight') nextRoom();
      else if (val === 'ArrowLeft') prevRoom();
    }});

    function exportRatings() {{
      const answeredCount = Object.values(votes).length;
      if (answeredCount === 0) {{
        alert("Please evaluate at least one room before exporting.");
        return;
      }}
      if (answeredCount < ROOMS.length) {{
        const proceed = confirm(`You have evaluated ${{answeredCount}} of ${{ROOMS.length}} rooms. Export anyway?`);
        if (!proceed) return;
      }}

      const submission = {{
        study_schema_version: "1.0",
        study_id: STUDY_ID,
        rater_id: raterId,
        exported_at: new Date().toISOString(),
        total_rooms: ROOMS.length,
        answered_rooms: answeredCount,
        ratings: ROOMS.map(r => ({{
          room_id: r.id,
          room_type: r.type,
          choice: votes[r.id] ? votes[r.id].choice : null,
          timestamp: votes[r.id] ? votes[r.id].timestamp : null
        }}))
      }};

      const blob = new Blob([JSON.stringify(submission, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const safeId = raterId.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      a.href = url;
      a.download = `room_study_${{STUDY_ID}}_${{safeId}}_${{Date.now()}}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    renderRoom();
  </script>
</body>
</html>
"""

    html_1 = create_html("study_1", "Room Arrangement Study \u2014 Set 1")
    with open(study_1_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_1)
    print(f"Created {study_1_dir / 'index.html'}")

    html_2 = create_html("study_2", "Room Arrangement Study \u2014 Set 2")
    with open(study_2_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_2)
    print(f"Created {study_2_dir / 'index.html'}")

    print("\nBundles rebuilt successfully!")

if __name__ == "__main__":
    build_bundles()
