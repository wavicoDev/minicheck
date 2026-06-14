import json
import os
from pathlib import Path
from typing import Iterator, Tuple, List

DATA_ROOT = Path(r"C:\Users\MSI\Projects\minicheck\데이터셋\1.Training\라벨링데이터\TL1")


def iter_samples(limit: int = None) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (file_id, doc, claim) tuples from TL1 dataset.
    Each file can produce up to 3 samples (summary1/2/3).
    """
    count = 0
    # 정렬해서 순서 보장 (재실행 시 동일 순서)
    for json_path in sorted(DATA_ROOT.rglob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc = data["Meta(Refine)"]["passage"]
            file_id = data["Meta(Refine)"]["passage_id"]

            for key in ["summary1", "summary2", "summary3"]:
                claim = data["Annotation"].get(key)
                if claim:
                    yield (f"{file_id}_{key}", doc, claim)
                    count += 1
                    if limit and count >= limit:
                        return
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue


def load_batch(start_idx: int, batch_size: int) -> Tuple[List[str], List[str], List[str]]:
    """Load a specific batch by skipping start_idx samples."""
    ids, docs, claims = [], [], []
    for i, (fid, doc, claim) in enumerate(iter_samples()):
        if i < start_idx:
            continue
        if i >= start_idx + batch_size:
            break
        ids.append(fid)
        docs.append(doc)
        claims.append(claim)
    return ids, docs, claims


def count_total() -> int:
    """Count total samples (slow, use once)."""
    return sum(1 for _ in iter_samples())


if __name__ == "__main__":
    # Quick test
    for i, (fid, doc, claim) in enumerate(iter_samples(limit=3)):
        print(f"[{i}] {fid}")
        print(f"  doc: {doc[:80]}...")
        print(f"  claim: {claim[:80]}...")
