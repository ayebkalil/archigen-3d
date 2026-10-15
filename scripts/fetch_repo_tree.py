import urllib.request
import json

url = "https://api.github.com/repos/CubiCasa/CubiCasa5k/git/trees/master?recursive=1"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Total files in repo:", len(data.get("tree", [])))
        for item in data.get("tree", []):
            path = item["path"]
            if any(k in path.lower() for k in ["loader", "svg", "floorplan", "dataset"]):
                print(" ", path)
except Exception as e:
    print("Error:", e)
