"""
TL1 데이터셋 평가 스크립트 (체크포인트 지원)
- 중간 저장: results/ 디렉토리에 JSON으로 저장
- 재시작 시 마지막 체크포인트부터 이어서 실행
"""
import json
import os
from pathlib import Path
from datetime import datetime

# GPU 상태 초기화 (다른 import 전에!)
import torch
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("[GPU] CUDA reset complete")

from data_loader import iter_samples

# test.py에서 scorer import
from test import RetrievalNLIScorer, CascadeVerifier

CHECKPOINT_DIR = Path("results")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"
RESULTS_FILE = CHECKPOINT_DIR / "eval_results.jsonl"

BATCH_SIZE = 100  # 몇 개마다 저장할지


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "correct": 0}


def load_processed_ids() -> tuple:
    """이미 처리된 샘플 ID 집합 및 correct 수 로드 (error 제외)"""
    ids = set()
    correct = 0
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # error 상태는 재처리 대상
                    if d.get("stage") != "error":
                        ids.add(d["id"])
                        if d.get("correct"):
                            correct += 1
                except:
                    continue
    return ids, correct


def save_checkpoint(state: dict):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def append_results(results: list):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_eval(limit: int = None):
    # 모델 로드
    path = r".\klue-roberta-nli"
    emb_path = r".\embedding"
    scorer = RetrievalNLIScorer(path, emb_path)
    verifier = CascadeVerifier(scorer, entail_th=0.5, contra_th=0.5)

    # 이미 처리된 ID 로드 (error 제외)
    processed_ids, correct = load_processed_ids()

    print(f"Already processed: {len(processed_ids)}, correct so far: {correct}")

    batch_ids, batch_docs, batch_claims = [], [], []
    batch_results = []
    processed_count = len(processed_ids)

    for i, (fid, doc, claim) in enumerate(iter_samples(limit=limit)):
        if fid in processed_ids:
            continue

        batch_ids.append(fid)
        batch_docs.append(doc)
        batch_claims.append(claim)

        # 배치가 찼으면 처리 (try-except 제거 - 날것 에러 확인용)
        if len(batch_docs) >= BATCH_SIZE:
            verdicts, metas = verifier.verify(batch_docs, batch_claims)

            for j, (bid, pred, meta) in enumerate(zip(batch_ids, verdicts, metas)):
                gold = 1
                is_correct = (pred == gold)
                correct += is_correct
                processed_count += 1

                batch_results.append({
                    "id": bid,
                    "pred": pred,
                    "stage": meta.get("stage"),
                    "gold": gold,
                    "correct": is_correct
                })

            # 저장
            append_results(batch_results)
            state = {
                "processed": processed_count,
                "correct": correct,
                "timestamp": datetime.now().isoformat()
            }
            save_checkpoint(state)

            acc = correct / processed_count * 100
            print(f"[{processed_count}] acc={acc:.2f}% ({correct}/{processed_count})")

            batch_ids, batch_docs, batch_claims = [], [], []
            batch_results = []

            # 주기적 메모리 정리
            if processed_count % 500 == 0:
                import torch
                import gc
                torch.cuda.empty_cache()
                scorer.clear_cache()
                gc.collect()
                print(f"  [MEM] cache cleared at {processed_count}")

    # 남은 배치 처리
    if batch_docs:
        try:
            verdicts, metas = verifier.verify(batch_docs, batch_claims)
        except Exception as e:
            print(f"[ERROR] Final batch failed: {e}")
            verdicts = [0] * len(batch_docs)
            metas = [{"stage": "error"}] * len(batch_docs)
        for bid, pred, meta in zip(batch_ids, verdicts, metas):
            gold = 1
            correct += (pred == gold)
            processed_count += 1
            batch_results.append({
                "id": bid,
                "pred": pred,
                "stage": meta.get("stage"),
                "gold": gold,
                "correct": pred == gold
            })
        append_results(batch_results)
        state = {
            "processed": processed_count,
            "correct": correct,
            "timestamp": datetime.now().isoformat()
        }
        save_checkpoint(state)

    if processed_count > 0:
        print(f"\nDone! Final accuracy: {correct}/{processed_count} = {correct/processed_count*100:.2f}%")
    else:
        print("\nNo new samples to process.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")
    args = parser.parse_args()

    if args.reset:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
        print("Checkpoint reset.")

    run_eval(limit=args.limit)
