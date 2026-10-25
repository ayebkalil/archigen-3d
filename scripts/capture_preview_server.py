"""
scripts/capture_preview_server.py
---------------------------------
Captures a pixel-perfect 1920x1080 screenshot of the furniture preview scene
directly from Three.js WebGL canvas once all models are loaded.
"""

import os
import sys
import json
import base64
import time
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8098
OUTPUT_PATH = r"C:\Users\ayebk\.gemini\antigravity\brain\387c7b87-863d-4c00-831b-93abdc64d4d4\furniture_bank_preview.png"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "public"))

class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/save_screenshot":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode("utf-8"))
            data_url = payload.get("image", "")
            
            if "," in data_url:
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
                
            img_bytes = base64.b64decode(encoded)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "wb") as f:
                f.write(img_bytes)
                
            print(f"[SERVER] Captured screenshot written to {OUTPUT_PATH} ({len(img_bytes)} bytes)")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            self.server.captured = True
        else:
            self.send_response(404)
            self.end_headers()

def run():
    server = HTTPServer(("127.0.0.1", PORT), PreviewHandler)
    server.captured = False
    print(f"[SERVER] Serving {BASE_DIR} on http://127.0.0.1:{PORT}")
    
    # Launch browser
    browser_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(browser_exe):
        browser_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--window-size=1920,1080",
        f"http://127.0.0.1:{PORT}/preview_furniture.html"
    ]
    print(f"[SERVER] Launching browser: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    
    # Wait for capture
    start_time = time.time()
    while not server.captured and (time.time() - start_time) < 30:
        server.handle_request()
        
    proc.terminate()
    server.server_close()
    
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 50000:
        print(f"[SUCCESS] Preview screenshot generated: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)")
    else:
        print("[ERROR] Screenshot capture failed or timed out.")
        sys.exit(1)

if __name__ == "__main__":
    run()
