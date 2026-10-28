import os
from pathlib import Path

def byte_scan():
    base_dir = Path("outputs/blind_study/shared_bundles")
    search_words = ["engine", "random", "baseline", "b1", "b2", "key", "seed", "ground_truth"]

    print("=== RAW BYTE-LEVEL SCAN OF ALL FILES IN SHARED BUNDLES ===")
    matches = []

    for study_id in ["study_1", "study_2"]:
        s_dir = base_dir / study_id
        for root, dirs, files in os.walk(s_dir):
            for fname in files:
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(base_dir)

                # Check filename
                fname_lower = fname.lower()
                for word in search_words:
                    if word in fname_lower:
                        matches.append((str(rel_path), f"FILENAME contains '{word}'", 0))

                # Check byte content
                with open(fpath, "rb") as f:
                    content = f.read()

                content_lower = content.lower()
                for word in search_words:
                    word_bytes = word.encode("utf-8")
                    idx = 0
                    count = 0
                    while True:
                        idx = content_lower.find(word_bytes, idx)
                        if idx == -1:
                            break
                        count += 1
                        idx += len(word_bytes)

                    if count > 0:
                        matches.append((str(rel_path), f"CONTENT contains '{word}' ({count} times)", count))

    print(f"Total scan matches found: {len(matches)}")
    for m in matches:
        print(f"  {m[0]:<45} : {m[1]}")

if __name__ == "__main__":
    byte_scan()
