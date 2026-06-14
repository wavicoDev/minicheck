# -*- coding: utf-8 -*-
"""
1. _contra 오류 후보 추출 (CSV + JSONL)
2. LLM 검증용 포맷 생성
"""
import json
import csv
from pathlib import Path
from data_loader import iter_samples

RESULTS_DIR = Path("results")

def load_contra_ids():
    """_contra로 판정된 ID 목록 로드"""
    contra = []
    with open(RESULTS_DIR / "eval_results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "_contra" in r.get("stage", ""):
                contra.append({
                    "id": r["id"],
                    "stage": r["stage"],
                })
    return contra

def extract_contra_samples():
    """_contra 샘플 전체 추출"""
    contra_info = load_contra_ids()
    contra_ids = {c["id"]: c["stage"] for c in contra_info}

    print(f"_contra 샘플 수: {len(contra_ids)}")

    # 샘플 데이터 수집
    samples = []
    for fid, doc, claim in iter_samples():
        if fid in contra_ids:
            samples.append({
                "id": fid,
                "stage": contra_ids[fid],
                "doc": doc,
                "claim": claim,
            })
        if len(samples) == len(contra_ids):
            break

    print(f"수집 완료: {len(samples)}개")

    # CSV 저장
    csv_path = RESULTS_DIR / "contra_errors.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "stage", "claim", "doc"])
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s["id"],
                "stage": s["stage"],
                "claim": s["claim"],
                "doc": s["doc"][:2000],  # 문서 길이 제한
            })
    print(f"CSV 저장: {csv_path}")

    # JSONL 저장 (LLM 검증용)
    jsonl_path = RESULTS_DIR / "contra_errors.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            json.dump(s, f, ensure_ascii=False)
            f.write("\n")
    print(f"JSONL 저장: {jsonl_path}")

    return samples

if __name__ == "__main__":
    extract_contra_samples()
