# -*- coding: utf-8 -*-
"""
LLM 배치 검증 - 오류 후보 2,222개를 LLM으로 검증
지원: OpenAI, Anthropic, Ollama (로컬)
"""
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RESULTS_DIR = Path("results")

# ============ 프롬프트 템플릿 ============

SYSTEM_PROMPT = """당신은 팩트체크 전문가입니다.
주어진 문서(원문)와 요약문을 비교하여, 요약문이 문서의 내용을 정확하게 반영하는지 판단합니다.

판정 기준:
- SUPPORTED: 요약문의 모든 주장이 문서에서 직접 확인되거나 합리적으로 추론 가능
- UNSUPPORTED: 요약문에 문서에 없는 정보가 추가되었거나, 사실이 왜곡됨
- PARTIAL: 일부는 맞지만 일부는 틀리거나 확인 불가

오류 유형 (UNSUPPORTED 또는 PARTIAL인 경우):
- NUMBER: 숫자/수량 불일치 (예: 3명→5명, 10%→15%)
- DATE: 날짜/시간 불일치 (예: 2020년→2021년)
- NAME: 이름/고유명사 오류 (예: 김철수→김철호, DPAA→DAPP)
- FACT: 사실 왜곡 또는 반대로 서술
- HALLUCINATION: 문서에 전혀 없는 내용 추가
- OMISSION: 중요한 맥락 누락으로 의미 왜곡
- OTHER: 기타

반드시 아래 JSON 형식으로만 응답하세요:
```json
{
  "verdict": "SUPPORTED" | "UNSUPPORTED" | "PARTIAL",
  "error_type": null | "NUMBER" | "DATE" | "NAME" | "FACT" | "HALLUCINATION" | "OMISSION" | "OTHER",
  "confidence": 0.0~1.0,
  "reason": "판단 근거를 1-2문장으로 설명"
}
```"""

USER_PROMPT_TEMPLATE = """[문서]
{doc}

[요약문]
{claim}

위 요약문이 문서 내용을 정확하게 반영하는지 판단하세요."""


# ============ LLM 클라이언트 ============

@dataclass
class LLMConfig:
    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    model: str = "gpt-4o-mini"  # or "claude-3-haiku-20240307", "llama3"
    api_key: str = None
    base_url: str = None  # Ollama: "http://localhost:11434/v1"
    max_tokens: int = 500
    temperature: float = 0.0


def call_openai(config: LLMConfig, doc: str, claim: str) -> dict:
    """OpenAI API 호출"""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=config.base_url,
    )

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(doc=doc[:3000], claim=claim)},
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def call_anthropic(config: LLMConfig, doc: str, claim: str) -> dict:
    """Anthropic API 호출"""
    import anthropic

    client = anthropic.Anthropic(
        api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
    )

    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(doc=doc[:3000], claim=claim)},
        ],
    )

    # JSON 파싱
    text = response.content[0].text
    # ```json ... ``` 블록 추출
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


def call_ollama(config: LLMConfig, doc: str, claim: str) -> dict:
    """Ollama (로컬 LLM) 호출 - OpenAI 호환 API 사용"""
    config.base_url = config.base_url or "http://localhost:11434/v1"
    config.api_key = config.api_key or "ollama"  # dummy key
    return call_openai(config, doc, claim)


def call_llm(config: LLMConfig, doc: str, claim: str) -> dict:
    """LLM 호출 (provider에 따라 분기)"""
    if config.provider == "openai":
        return call_openai(config, doc, claim)
    elif config.provider == "anthropic":
        return call_anthropic(config, doc, claim)
    elif config.provider == "ollama":
        return call_ollama(config, doc, claim)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# ============ 배치 처리 ============

def verify_single(config: LLMConfig, sample: dict, retry: int = 3) -> dict:
    """단일 샘플 검증"""
    for attempt in range(retry):
        try:
            result = call_llm(config, sample["doc"], sample["claim"])
            return {
                "id": sample["id"],
                "stage": sample["stage"],
                "claim": sample["claim"],
                "llm_verdict": result.get("verdict"),
                "llm_error_type": result.get("error_type"),
                "llm_confidence": result.get("confidence"),
                "llm_reason": result.get("reason"),
            }
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                return {
                    "id": sample["id"],
                    "stage": sample["stage"],
                    "claim": sample["claim"],
                    "llm_verdict": "ERROR",
                    "llm_error_type": None,
                    "llm_confidence": None,
                    "llm_reason": str(e),
                }


def verify_batch(
    config: LLMConfig,
    input_path: Path = RESULTS_DIR / "contra_errors.jsonl",
    output_path: Path = RESULTS_DIR / "llm_verified.jsonl",
    limit: int = None,
    workers: int = 5,
    checkpoint_interval: int = 50,
):
    """배치 검증 실행"""
    # 입력 로드
    samples = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    if limit:
        samples = samples[:limit]

    print(f"검증 대상: {len(samples)}개")
    print(f"Provider: {config.provider}, Model: {config.model}")

    # 이미 처리된 ID 로드 (이어하기)
    processed_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                processed_ids.add(r["id"])
        print(f"이미 처리됨: {len(processed_ids)}개")

    # 미처리 샘플 필터
    pending = [s for s in samples if s["id"] not in processed_ids]
    print(f"처리 예정: {len(pending)}개")

    if not pending:
        print("모두 처리 완료!")
        return

    # 병렬 처리
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify_single, config, s): s for s in pending}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            # 진행 상황
            if (i + 1) % 10 == 0:
                done = len(processed_ids) + len(results)
                print(f"[{done}/{len(samples)}] {result['id']}: {result['llm_verdict']}")

            # 체크포인트 저장
            if (i + 1) % checkpoint_interval == 0:
                with open(output_path, "a", encoding="utf-8") as f:
                    for r in results:
                        json.dump(r, f, ensure_ascii=False)
                        f.write("\n")
                results = []

    # 남은 결과 저장
    if results:
        with open(output_path, "a", encoding="utf-8") as f:
            for r in results:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

    print(f"\n완료! 결과 저장: {output_path}")


def summarize_results(path: Path = RESULTS_DIR / "llm_verified.jsonl"):
    """결과 요약"""
    if not path.exists():
        print(f"파일 없음: {path}")
        return

    results = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"\n=== LLM 검증 결과 요약 ({len(results)}개) ===\n")

    # 판정 분포
    from collections import Counter
    verdicts = Counter(r["llm_verdict"] for r in results)
    print("판정 분포:")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c} ({100*c/len(results):.1f}%)")

    # 오류 유형 분포 (UNSUPPORTED/PARTIAL만)
    errors = [r for r in results if r["llm_verdict"] in ("UNSUPPORTED", "PARTIAL")]
    if errors:
        error_types = Counter(r["llm_error_type"] for r in errors)
        print(f"\n오류 유형 분포 ({len(errors)}개):")
        for t, c in error_types.most_common():
            print(f"  {t}: {c} ({100*c/len(errors):.1f}%)")

    # 샘플 출력
    print("\n--- UNSUPPORTED 샘플 5개 ---")
    unsupported = [r for r in results if r["llm_verdict"] == "UNSUPPORTED"][:5]
    for r in unsupported:
        print(f"\n[{r['id']}] {r['llm_error_type']}")
        print(f"  요약: {r['claim'][:100]}...")
        print(f"  이유: {r['llm_reason']}")


# ============ CLI ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 배치 검증")
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama"], default="openai")
    parser.add_argument("--model", default=None, help="모델명 (기본: gpt-4o-mini)")
    parser.add_argument("--limit", type=int, default=None, help="처리할 샘플 수 제한")
    parser.add_argument("--workers", type=int, default=5, help="병렬 처리 수")
    parser.add_argument("--summarize", action="store_true", help="결과 요약만 출력")
    args = parser.parse_args()

    if args.summarize:
        summarize_results()
    else:
        # 기본 모델 설정
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "ollama": "llama3",
        }

        config = LLMConfig(
            provider=args.provider,
            model=args.model or default_models[args.provider],
        )

        verify_batch(config, limit=args.limit, workers=args.workers)
        summarize_results()
