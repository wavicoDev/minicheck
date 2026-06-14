# -*- coding: utf-8 -*-
"""
오류 유형 자동 분류
- 숫자 불일치
- 이름/고유명사 오류
- 날짜 불일치
- 없는 내용 추가 (환각)
"""
import json
import re
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("results")

# 숫자 패턴 (한글 숫자 포함)
NUM_PATTERN = re.compile(r'\d+(?:\.\d+)?(?:%|원|명|개|건|억|만|천|백)?|[일이삼사오육칠팔구십백천만억조]+(?:명|원|개|건)?')

# 날짜 패턴
DATE_PATTERN = re.compile(r'\d{1,4}년|\d{1,2}월|\d{1,2}일|\d{1,2}시|\d{1,2}분')

# 고유명사 추출 (따옴표, 대문자 시작 단어 등)
PROPER_NOUN_PATTERN = re.compile(r"'[^']+?'|\"[^\"]+?\"|[A-Z][a-zA-Z]+|[가-힣]+(?:부|청|원|회|당|사|국|위원회|연구소|대학|병원)")


def extract_numbers(text):
    """텍스트에서 숫자 추출"""
    return set(NUM_PATTERN.findall(text))


def extract_dates(text):
    """텍스트에서 날짜 추출"""
    return set(DATE_PATTERN.findall(text))


def extract_proper_nouns(text):
    """텍스트에서 고유명사 추출"""
    return set(PROPER_NOUN_PATTERN.findall(text))


def classify_error(doc, claim):
    """
    오류 유형 분류
    Returns: list of error types
    """
    errors = []

    doc_nums = extract_numbers(doc)
    claim_nums = extract_numbers(claim)

    doc_dates = extract_dates(doc)
    claim_dates = extract_dates(claim)

    doc_nouns = extract_proper_nouns(doc)
    claim_nouns = extract_proper_nouns(claim)

    # 1. 숫자 불일치: 요약에 있는 숫자가 문서에 없음
    claim_only_nums = claim_nums - doc_nums
    if claim_only_nums:
        errors.append({
            "type": "number_mismatch",
            "detail": f"요약에만 있는 숫자: {list(claim_only_nums)[:5]}"
        })

    # 2. 날짜 불일치
    claim_only_dates = claim_dates - doc_dates
    if claim_only_dates:
        errors.append({
            "type": "date_mismatch",
            "detail": f"요약에만 있는 날짜: {list(claim_only_dates)[:5]}"
        })

    # 3. 고유명사 오류: 요약의 고유명사가 문서에 없음
    claim_only_nouns = claim_nouns - doc_nouns
    # 문서에 부분적으로 포함된 경우 제외
    real_missing = []
    for noun in claim_only_nouns:
        if len(noun) >= 3 and noun not in doc:
            real_missing.append(noun)
    if real_missing:
        errors.append({
            "type": "proper_noun_error",
            "detail": f"문서에 없는 고유명사: {real_missing[:5]}"
        })

    # 4. 환각 탐지: 요약 문장이 문서에 전혀 근거 없음
    # (간단 휴리스틱: 요약의 주요 키워드가 문서에 거의 없는 경우)
    claim_words = set(re.findall(r'[가-힣]{2,}', claim))
    doc_words = set(re.findall(r'[가-힣]{2,}', doc))
    overlap = claim_words & doc_words
    if len(claim_words) > 0:
        overlap_ratio = len(overlap) / len(claim_words)
        if overlap_ratio < 0.3:
            errors.append({
                "type": "hallucination",
                "detail": f"키워드 중복률 {overlap_ratio:.1%} (낮음)"
            })

    # 분류 안 됨
    if not errors:
        errors.append({
            "type": "unknown",
            "detail": "자동 분류 실패 - 수동 검토 필요"
        })

    return errors


def classify_all():
    """모든 _contra 샘플 분류"""
    input_path = RESULTS_DIR / "contra_errors.jsonl"
    if not input_path.exists():
        print("먼저 extract_errors.py를 실행하세요")
        return

    samples = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    print(f"분류할 샘플: {len(samples)}개")

    # 분류 실행
    results = []
    type_counter = Counter()

    for s in samples:
        errors = classify_error(s["doc"], s["claim"])
        for e in errors:
            type_counter[e["type"]] += 1

        results.append({
            "id": s["id"],
            "stage": s["stage"],
            "claim": s["claim"],
            "doc_preview": s["doc"][:300],
            "error_types": errors,
        })

    # 결과 저장
    output_path = RESULTS_DIR / "classified_errors.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"분류 결과 저장: {output_path}")

    # 통계 출력
    print(f"\n=== 오류 유형 분포 ===")
    for err_type, count in type_counter.most_common():
        pct = 100 * count / len(samples)
        print(f"  {err_type}: {count} ({pct:.1f}%)")

    # 유형별 샘플 저장
    save_samples_by_type(results)

    return results


def save_samples_by_type(results):
    """유형별로 샘플 10개씩 저장"""
    by_type = {}
    for r in results:
        for e in r["error_types"]:
            t = e["type"]
            if t not in by_type:
                by_type[t] = []
            if len(by_type[t]) < 10:
                by_type[t].append({
                    "id": r["id"],
                    "claim": r["claim"],
                    "doc_preview": r["doc_preview"],
                    "detail": e["detail"],
                })

    output_path = RESULTS_DIR / "error_samples_by_type.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(by_type, f, ensure_ascii=False, indent=2)
    print(f"유형별 샘플 저장: {output_path}")


if __name__ == "__main__":
    classify_all()
