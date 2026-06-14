# -*- coding: utf-8 -*-
"""
OpenAI Batch API를 사용한 대량 검증 (50% 비용 절감)
- 24시간 내 처리
- 대량 처리에 적합
"""
import json
import os
from pathlib import Path
from datetime import datetime

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 없으면 환경변수 직접 사용

RESULTS_DIR = Path("results")

SYSTEM_PROMPT = """당신은 팩트체크 전문가입니다.
주어진 문서(원문)와 요약문을 비교하여, 요약문이 문서의 내용을 정확하게 반영하는지 판단합니다.

판정 기준:
- SUPPORTED: 요약문의 모든 주장이 문서에서 직접 확인되거나 합리적으로 추론 가능
- UNSUPPORTED: 요약문에 문서에 없는 정보가 추가되었거나, 사실이 왜곡됨
- PARTIAL: 일부는 맞지만 일부는 틀리거나 확인 불가

오류 유형 (UNSUPPORTED 또는 PARTIAL인 경우):
- NUMBER: 숫자/수량 불일치
- DATE: 날짜/시간 불일치
- NAME: 이름/고유명사 오류
- FACT: 사실 왜곡 또는 반대로 서술
- HALLUCINATION: 문서에 전혀 없는 내용 추가
- OMISSION: 중요한 맥락 누락으로 의미 왜곡
- OTHER: 기타

반드시 아래 JSON 형식으로만 응답:
{"verdict": "SUPPORTED|UNSUPPORTED|PARTIAL", "error_type": null|"TYPE", "confidence": 0.0~1.0, "reason": "판단 근거"}"""


def create_batch_file(
    input_path: Path = RESULTS_DIR / "contra_errors.jsonl",
    output_path: Path = RESULTS_DIR / "batch_requests.jsonl",
    model: str = "gpt-4o-mini",
    limit: int = None,
):
    """Batch API용 JSONL 파일 생성"""
    samples = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    if limit:
        samples = samples[:limit]

    print(f"배치 파일 생성: {len(samples)}개")

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            request = {
                "custom_id": s["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"[문서]\n{s['doc'][:3000]}\n\n[요약문]\n{s['claim']}"},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            }
            json.dump(request, f, ensure_ascii=False)
            f.write("\n")

    print(f"저장: {output_path}")
    print(f"파일 크기: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return output_path


def upload_and_submit(file_path: Path):
    """배치 작업 제출"""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 파일 업로드
    print("파일 업로드 중...")
    with open(file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    print(f"업로드 완료: {batch_file.id}")

    # 배치 작업 생성
    print("배치 작업 제출 중...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"minicheck verification {datetime.now().isoformat()}"},
    )
    print(f"배치 ID: {batch.id}")
    print(f"상태: {batch.status}")

    # 배치 ID 저장
    info_path = RESULTS_DIR / "batch_info.json"
    with open(info_path, "w") as f:
        json.dump({"batch_id": batch.id, "file_id": batch_file.id, "created": datetime.now().isoformat()}, f)
    print(f"배치 정보 저장: {info_path}")

    return batch.id


def check_status(batch_id: str = None):
    """배치 상태 확인"""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not batch_id:
        info_path = RESULTS_DIR / "batch_info.json"
        if info_path.exists():
            batch_id = json.load(open(info_path))["batch_id"]
        else:
            print("batch_id를 지정하거나 batch_info.json이 필요합니다")
            return

    batch = client.batches.retrieve(batch_id)
    print(f"배치 ID: {batch.id}")
    print(f"상태: {batch.status}")
    print(f"진행: {batch.request_counts.completed}/{batch.request_counts.total}")

    if batch.status == "completed":
        print(f"출력 파일 ID: {batch.output_file_id}")
    elif batch.status == "failed":
        print(f"에러: {batch.errors}")

    return batch


def download_results(batch_id: str = None, output_path: Path = RESULTS_DIR / "llm_verified.jsonl"):
    """완료된 배치 결과 다운로드"""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not batch_id:
        info_path = RESULTS_DIR / "batch_info.json"
        batch_id = json.load(open(info_path))["batch_id"]

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"배치가 아직 완료되지 않음: {batch.status}")
        return

    # 결과 다운로드
    print(f"결과 다운로드 중... (file_id: {batch.output_file_id})")
    content = client.files.content(batch.output_file_id)

    # 파싱 및 저장
    results = []
    for line in content.text.strip().split("\n"):
        resp = json.loads(line)
        custom_id = resp["custom_id"]

        if resp["response"]["status_code"] == 200:
            body = resp["response"]["body"]
            llm_output = json.loads(body["choices"][0]["message"]["content"])
            results.append({
                "id": custom_id,
                "llm_verdict": llm_output.get("verdict"),
                "llm_error_type": llm_output.get("error_type"),
                "llm_confidence": llm_output.get("confidence"),
                "llm_reason": llm_output.get("reason"),
            })
        else:
            results.append({
                "id": custom_id,
                "llm_verdict": "ERROR",
                "llm_error_type": None,
                "llm_confidence": None,
                "llm_reason": str(resp["response"]),
            })

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    print(f"저장 완료: {output_path} ({len(results)}개)")

    # 요약
    from collections import Counter
    verdicts = Counter(r["llm_verdict"] for r in results)
    print("\n=== 결과 요약 ===")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c} ({100*c/len(results):.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI Batch API 검증")
    parser.add_argument("action", choices=["create", "submit", "status", "download"],
                        help="create: 배치 파일 생성, submit: 제출, status: 상태 확인, download: 결과 다운로드")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()

    if args.action == "create":
        create_batch_file(model=args.model, limit=args.limit)
    elif args.action == "submit":
        batch_path = RESULTS_DIR / "batch_requests.jsonl"
        if not batch_path.exists():
            print("먼저 'create' 명령으로 배치 파일을 생성하세요")
        else:
            upload_and_submit(batch_path)
    elif args.action == "status":
        check_status(args.batch_id)
    elif args.action == "download":
        download_results(args.batch_id)
