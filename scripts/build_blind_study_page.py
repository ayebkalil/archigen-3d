import json
from pathlib import Path

def build_page():
    manifest_path = Path("outputs/blind_study/study_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_json_str = json.dumps(manifest, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ArchiGen 3D — Room Design Blind Evaluation</title>
  <style>
    :root {{
      --bg: #0b1120;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --selected: #0284c7;
      --success: #10b981;
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
      background: #0f172a;
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .logo {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 16px;
      letter-spacing: 0.5px;
    }}
    .logo span {{ color: var(--accent); }}
    .header-controls {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    .rater-box {{
      display: flex;
      align-items: center;
      gap: 8px;
      background: #1e293b;
      border: 1px solid var(--card-border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .rater-box input {{
      background: transparent;
      border: none;
      color: #fff;
      font-weight: 600;
      outline: none;
      width: 140px;
    }}
    .progress-bar-container {{
      width: 180px;
      height: 8px;
      background: #334155;
      border-radius: 4px;
      overflow: hidden;
    }}
    .progress-bar-fill {{
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 0.3s ease;
    }}
    .btn {{
      background: #334155;
      color: #fff;
      border: 1px solid #475569;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn:hover {{ background: #475569; }}
    .btn-primary {{ background: var(--accent); color: #0b1120; border-color: var(--accent); }}
    .btn-primary:hover {{ background: var(--accent-hover); color: #fff; }}
    .btn-success {{ background: var(--success); color: #fff; border-color: var(--success); }}
    .btn-success:hover {{ background: #059669; }}

    main {{
      flex: 1;
      padding: 20px 24px;
      max-width: 1500px;
      margin: 0 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    .room-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 12px 20px;
      border-radius: 8px;
    }}
    .room-title {{
      font-size: 18px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .badge {{
      background: #0369a1;
      color: #e0f2fe;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .room-meta {{
      font-size: 13px;
      color: var(--text-muted);
    }}

    .comparison-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      flex: 1;
    }}
    .image-card {{
      background: var(--card-bg);
      border: 2px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }}
    .image-card:hover {{
      border-color: #64748b;
    }}
    .image-card.selected {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.35);
    }}
    .image-card-header {{
      background: rgba(15, 23, 42, 0.85);
      padding: 10px 16px;
      font-size: 14px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
    }}
    .image-container {{
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #020617;
      min-height: 440px;
      position: relative;
    }}
    .image-container img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      user-select: none;
    }}

    .decision-bar {{
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 16px;
      padding: 12px 0;
    }}
    .vote-btn {{
      padding: 14px 28px;
      border-radius: 8px;
      font-size: 15px;
      font-weight: 700;
      border: 2px solid var(--card-border);
      background: #1e293b;
      color: #fff;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .vote-btn:hover {{
      background: #334155;
      border-color: #64748b;
    }}
    .vote-btn.active {{
      background: #0284c7;
      border-color: var(--accent);
      color: #fff;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
    }}
    .vote-btn .key-hint {{
      font-size: 11px;
      background: rgba(0,0,0,0.3);
      padding: 2px 6px;
      border-radius: 4px;
      color: #cbd5e1;
    }}

    .footer-nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 0;
    }}
    .room-navigator {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: center;
      max-width: 800px;
    }}
    .nav-dot {{
      width: 26px;
      height: 26px;
      border-radius: 4px;
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .nav-dot:hover {{ background: #334155; color: #fff; }}
    .nav-dot.voted {{ background: #065f46; border-color: #059669; color: #a7f3d0; }}
    .nav-dot.current {{ border-color: var(--accent); color: #fff; font-weight: bold; box-shadow: 0 0 6px var(--accent); }}

    /* Modal for instructions */
    .modal-overlay {{
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.75);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s;
    }}
    .modal-overlay.open {{ opacity: 1; pointer-events: auto; }}
    .modal {{
      background: #1e293b;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      max-width: 640px;
      width: 90%;
      padding: 28px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      line-height: 1.6;
    }}
    .modal h2 {{ font-size: 20px; margin-bottom: 12px; color: var(--accent); }}
    .modal p {{ color: #cbd5e1; font-size: 14px; margin-bottom: 12px; }}
    .modal ul {{ margin-left: 20px; margin-bottom: 18px; color: #cbd5e1; font-size: 14px; }}
    .modal li {{ margin-bottom: 6px; }}

    /* Complete Banner */
    .complete-banner {{
      display: none;
      background: linear-gradient(135deg, #065f46, #047857);
      border: 1px solid #10b981;
      padding: 24px;
      border-radius: 12px;
      text-align: center;
      margin: 20px 0;
    }}
    .complete-banner.show {{ display: block; }}
    .complete-banner h3 {{ font-size: 22px; margin-bottom: 8px; color: #fff; }}
    .complete-banner p {{ color: #d1fae5; font-size: 14px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <span>ARCHIGEN 3D</span> | Blind Human Evaluation
    </div>
    <div class="header-controls">
      <div class="rater-box">
        <span style="color: var(--text-muted);">Rater ID:</span>
        <input type="text" id="rater-id-input" placeholder="e.g. rater_01" value="rater_01">
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span id="progress-text" style="font-size: 12px; color: var(--text-muted); font-weight: 600;">0 / 32</span>
        <div class="progress-bar-container">
          <div class="progress-bar-fill" id="progress-bar-fill"></div>
        </div>
      </div>
      <button class="btn" id="btn-instructions">Instructions</button>
      <button class="btn btn-success" id="btn-export-top" style="display: none;">Export JSON</button>
    </div>
  </header>

  <main>
    <div class="complete-banner" id="complete-banner">
      <h3>🎉 All 32 Rooms Evaluated!</h3>
      <p>Thank you for completing this architectural layout evaluation. Please click below to download your verified ratings file.</p>
      <button class="btn btn-success" id="btn-export-main" style="font-size: 16px; padding: 12px 24px;">
        💾 Export Ratings JSON
      </button>
    </div>

    <div class="room-header">
      <div class="room-title">
        <span id="room-number-label">Room 1 of 32</span>
        <span class="badge" id="room-type-badge">BEDROOM</span>
      </div>
      <div class="room-meta">
        Floor Area: <strong id="room-area-label" style="color: #fff;">13.22 m²</strong> &bull;
        Review furniture plausibility, circulation, and functional placement
      </div>
    </div>

    <div class="comparison-grid">
      <div class="image-card" id="card-left" onclick="castVote('Left')">
        <div class="image-card-header">
          <span>Option A (Left)</span>
          <span style="font-size: 11px; color: var(--text-muted); font-weight: normal;">Shortcut: [1] or [A]</span>
        </div>
        <div class="image-container">
          <img id="img-left" src="" alt="Option A">
        </div>
      </div>

      <div class="image-card" id="card-right" onclick="castVote('Right')">
        <div class="image-card-header">
          <span>Option B (Right)</span>
          <span style="font-size: 11px; color: var(--text-muted); font-weight: normal;">Shortcut: [3] or [B]</span>
        </div>
        <div class="image-container">
          <img id="img-right" src="" alt="Option B">
        </div>
      </div>
    </div>

    <div class="decision-bar">
      <button class="vote-btn" id="vote-btn-left" onclick="castVote('Left')">
        <span>👈 Prefer Left (A)</span>
        <span class="key-hint">1</span>
      </button>

      <button class="vote-btn" id="vote-btn-tie" onclick="castVote('Tie')">
        <span>⚖️ Tie / Equivalent</span>
        <span class="key-hint">2</span>
      </button>

      <button class="vote-btn" id="vote-btn-right" onclick="castVote('Right')">
        <span>Prefer Right (B) 👉</span>
        <span class="key-hint">3</span>
      </button>
    </div>

    <div class="footer-nav">
      <button class="btn" id="btn-prev" onclick="prevRoom()">← Previous</button>
      <div class="room-navigator" id="room-navigator"></div>
      <button class="btn btn-primary" id="btn-next" onclick="nextRoom()">Next →</button>
    </div>
  </main>

  <div class="modal-overlay" id="modal-overlay" onclick="closeInstructions()">
    <div class="modal" onclick="event.stopPropagation()">
      <h2>Architectural Layout Evaluation Instructions</h2>
      <p>You will evaluate <strong>32 residential room layouts</strong>. In each case, two candidate 3D furniture arrangements are presented for the identical architectural room polygon with identical furniture assets.</p>
      <p><strong>What to look for:</strong></p>
      <ul>
        <li><strong>Circulation & Door Access:</strong> Can a person enter through the door? Are primary walking paths clear or blocked by misplaced items?</li>
        <li><strong>Placement Plausibility:</strong> Are large items (beds, sofas, counters, wardrobes) logically backed against walls? Do bedside tables flank the bed?</li>
        <li><strong>Physical Feasibility:</strong> Are pieces colliding or intersecting each other? Are items rotated arbitrarily?</li>
      </ul>
      <p><strong>How to vote:</strong> Choose <strong>Left</strong> or <strong>Right</strong> based on which layout is better. Select <strong>Tie</strong> if both layouts are equally good or equally bad. Use keyboard shortcuts (<code>1</code>: Left, <code>2</code>: Tie, <code>3</code>: Right) for fast evaluation.</p>
      <div style="text-align: right; margin-top: 20px;">
        <button class="btn btn-primary" onclick="closeInstructions()">Got it, Start Evaluating</button>
      </div>
    </div>
  </div>

  <script>
    const MANIFEST = {manifest_json_str};
    const ROOMS = MANIFEST.rooms;
    let currentIndex = 0;
    const votes = {{}}; // {{ room_id: {{ choice: 'Left'|'Right'|'Tie', timestamp: ISOString }} }}

    function init() {{
      renderNavDots();
      loadRoom(0);

      document.getElementById('btn-instructions').addEventListener('click', openInstructions);
      document.getElementById('btn-export-top').addEventListener('click', exportRatingsJSON);
      document.getElementById('btn-export-main').addEventListener('click', exportRatingsJSON);

      // Keyboard navigation
      window.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT') return;
        if (e.key === '1' || e.key === 'a' || e.key === 'A') castVote('Left');
        else if (e.key === '2' || e.key === 't' || e.key === 'T' || e.key === ' ') castVote('Tie');
        else if (e.key === '3' || e.key === 'b' || e.key === 'B') castVote('Right');
        else if (e.key === 'ArrowLeft') prevRoom();
        else if (e.key === 'ArrowRight') nextRoom();
      }});
    }}

    function renderNavDots() {{
      const container = document.getElementById('room-navigator');
      container.innerHTML = '';
      ROOMS.forEach((r, idx) => {{
        const dot = document.createElement('div');
        dot.className = 'nav-dot' + (idx === currentIndex ? ' current' : '') + (votes[r.room_id] ? ' voted' : '');
        dot.innerText = idx + 1;
        dot.title = `${{r.room_id}} (${{r.room_type}})`;
        dot.onclick = () => loadRoom(idx);
        container.appendChild(dot);
      }});
    }}

    function loadRoom(idx) {{
      if (idx < 0 || idx >= ROOMS.length) return;
      currentIndex = idx;
      const r = ROOMS[idx];

      document.getElementById('room-number-label').innerText = `Room ${{idx + 1}} of ${{ROOMS.length}}`;
      document.getElementById('room-type-badge').innerText = r.room_type.replace('_', ' ');
      document.getElementById('room-area-label').innerText = `${{r.area_m2}} m²`;

      document.getElementById('img-left').src = r.image_left;
      document.getElementById('img-right').src = r.image_right;

      document.getElementById('btn-prev').disabled = (idx === 0);
      document.getElementById('btn-next').innerText = (idx === ROOMS.length - 1) ? 'Finish' : 'Next →';

      updateVoteButtonsUI(r.room_id);
      renderNavDots();
      updateProgress();
    }}

    function castVote(choice) {{
      const r = ROOMS[currentIndex];
      votes[r.room_id] = {{
        room_id: r.room_id,
        room_type: r.room_type,
        choice: choice,
        timestamp: new Date().toISOString()
      }};
      updateVoteButtonsUI(r.room_id);
      renderNavDots();
      updateProgress();

      // Auto-advance to next room if not last room
      if (currentIndex < ROOMS.length - 1) {{
        setTimeout(() => {{
          loadRoom(currentIndex + 1);
        }}, 200);
      }} else {{
        checkCompletion();
      }}
    }}

    function updateVoteButtonsUI(roomId) {{
      const v = votes[roomId]?.choice;
      document.getElementById('vote-btn-left').className = 'vote-btn' + (v === 'Left' ? ' active' : '');
      document.getElementById('vote-btn-tie').className = 'vote-btn' + (v === 'Tie' ? ' active' : '');
      document.getElementById('vote-btn-right').className = 'vote-btn' + (v === 'Right' ? ' active' : '');

      document.getElementById('card-left').className = 'image-card' + (v === 'Left' ? ' selected' : '');
      document.getElementById('card-right').className = 'image-card' + (v === 'Right' ? ' selected' : '');
    }}

    function prevRoom() {{
      if (currentIndex > 0) loadRoom(currentIndex - 1);
    }}

    function nextRoom() {{
      if (currentIndex < ROOMS.length - 1) loadRoom(currentIndex + 1);
      else checkCompletion();
    }}

    function updateProgress() {{
      const votedCount = Object.keys(votes).length;
      const pct = Math.round((votedCount / ROOMS.length) * 100);
      document.getElementById('progress-text').innerText = `${{votedCount}} / ${{ROOMS.length}}`;
      document.getElementById('progress-bar-fill').style.width = `${{pct}}%`;

      if (votedCount === ROOMS.length) {{
        document.getElementById('btn-export-top').style.display = 'inline-flex';
        document.getElementById('complete-banner').className = 'complete-banner show';
      }}
    }}

    function checkCompletion() {{
      const votedCount = Object.keys(votes).length;
      if (votedCount === ROOMS.length) {{
        document.getElementById('complete-banner').className = 'complete-banner show';
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
    }}

    function openInstructions() {{
      document.getElementById('modal-overlay').className = 'modal-overlay open';
    }}

    function closeInstructions() {{
      document.getElementById('modal-overlay').className = 'modal-overlay';
    }}

    function exportRatingsJSON() {{
      const raterId = (document.getElementById('rater-id-input').value.trim()) || 'anonymous_rater';
      const votedList = ROOMS.map(r => votes[r.room_id] || {{ room_id: r.room_id, room_type: r.room_type, choice: 'Tie', timestamp: new Date().toISOString() }});

      const payload = {{
        archigen_study_export_version: "1.0",
        study_type: "room_design_blind_comparison",
        study_id: "cubicasa_stage5_blind_eval",
        exported_at: new Date().toISOString(),
        rater_id: raterId,
        total_rooms: ROOMS.length,
        completed_rooms: Object.keys(votes).length,
        results: votedList
      }};

      const jsonStr = JSON.stringify(payload, null, 2);
      const blob = new Blob([jsonStr], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const cleanRater = raterId.replace(/[^a-zA-Z0-9_-]/g, '_');
      a.href = url;
      a.download = `ratings_${{cleanRater}}_${{Date.now()}}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""
    out_file = Path("outputs/blind_study/index.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Built offline single static HTML page: {out_file} ({len(html_content)} bytes)")

if __name__ == "__main__":
    build_page()
