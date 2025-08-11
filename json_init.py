import os
import json
from typing import List, Dict, Any
from config import FILES_DIR, JSON_PATH


def json_init(output_json_path: str) -> None:
    """Scan Archive/ and dump sentences.json with raw sentence + source path:line."""
    all_sentences: List[Dict[str, Any]] = []
    print(f"Scanning '{FILES_DIR}' to create '{output_json_path}'...")
    for root, _, files in os.walk(FILES_DIR):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_number, sentence in enumerate(f, 1):
                        s = sentence.strip()
                        if s:
                            all_sentences.append(
                                {"sentence": s, "source": f"{file_path}:{line_number}"}
                            )
            except Exception as e:
                print(f"[WARN] {file_path}: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_sentences)} sentences to {output_json_path}.")


if __name__ == "__main__":
    json_init(JSON_PATH)
    print("JSON initialization complete.")
