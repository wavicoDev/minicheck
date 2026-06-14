import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


import torch
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi

class RetrievalNLIScorer:
    def __init__(self, nli_path, emb_path, top_k=5, threshold=0.5, batch_size=32, window=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 판사: NLI (기존과 동일)
        self.tok = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
        self.nli = AutoModelForSequenceClassification.from_pretrained(
            nli_path, local_files_only=True).eval().to(self.device)
        # 문지기: 임베딩 모델
        self.emb = SentenceTransformer(emb_path, local_files_only=True, device=self.device)
        self.emb.max_seq_length = 512  # position embedding 한계 맞춤
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size = batch_size
        self.entail_idx, self.contra_idx = 0, 2
        self.kiwi = Kiwi()
        self.window = window
        self._doc_cache = {}   # doc → (문장 리스트, 임베딩) 캐시
        self._chunk_cache = {}

    def _sentences(self, doc):
        if doc not in self._doc_cache:
            sents = [s.text for s in self.kiwi.split_into_sents(doc)]
            if not sents:
                sents = [doc[:500]] if doc else [""]
            # 임베딩 호출 전 검증
            try:
                embs = self.emb.encode(sents, normalize_embeddings=True)
            except Exception as e:
                print(f"[GUARD] 임베딩 실패: {e}")
                print(f"  sents 개수: {len(sents)}, 첫 문장: {sents[0][:50] if sents else 'N/A'}")
                embs = self.emb.encode([""], normalize_embeddings=True)
                embs = embs.repeat(len(sents), 1) if len(sents) > 1 else embs
            self._doc_cache[doc] = (sents, embs)
        return self._doc_cache[doc]

    def _chunks(self, doc):
        if doc in self._chunk_cache:
            return self._chunk_cache[doc]
        sents = [s.text for s in self.kiwi.split_into_sents(doc)]
        chunks = [doc] if len(sents) <= self.window else \
                 [" ".join(sents[i:i + self.window]) for i in range(len(sents) - self.window + 1)]
        print(f"--- {len(sents)}문장 → {len(chunks)}청크")   # 이제 doc당 1회만 출력
        self._chunk_cache[doc] = chunks
        return chunks

    def _retrieve(self, doc, claim, window=1):
        sents, sent_embs = self._sentences(doc)
        if len(sents) <= self.top_k:
            if window == 1:
                return sents
            else:
                return [" ".join(sents)]  # 짧으면 전체를 하나로

        c = self.emb.encode([claim], normalize_embeddings=True)[0]
        sims = sent_embs @ c                   # 코사인 유사도
        top_idx = sims.argsort()[-self.top_k:]

        if window == 1:
            return [sents[i] for i in top_idx]
        else:
            # window > 1: 각 top-k 문장 주변으로 window 확장
            results = []
            for i in top_idx:
                start = max(0, i - window // 2)
                end = min(len(sents), i + window // 2 + 1)
                results.append(" ".join(sents[start:end]))
            return results

    def clear_cache(self):
        """메모리 누적 방지용 캐시 정리"""
        self._doc_cache.clear()
        self._chunk_cache.clear()

    @torch.no_grad()
    def score(self, docs, claims, window=1, combine=False, entail_th=None, contra_th=None):
        """
        window: 검색된 문장 주변 몇 문장을 포함할지 (1=단일문장, 3=앞뒤 1문장씩)
        combine: True면 top-k 문장을 하나로 합쳐서 NLI (multi-hop용)
        entail_th: entailment 임계값 (None이면 self.threshold 사용)
        contra_th: contradiction 임계값 (기본 0.5)
        """
        entail_th = entail_th if entail_th is not None else self.threshold
        contra_th = contra_th if contra_th is not None else 0.5

        pairs, owner = [], []
        for i, (doc, claim) in enumerate(zip(docs, claims)):
            retrieved = self._retrieve(doc, claim, window=window)
            if not retrieved:
                pairs.append(("", claim))
                owner.append(i)
            elif combine:
                # top-k 문장을 하나로 합침
                combined = " ".join(retrieved)
                pairs.append((combined, claim))
                owner.append(i)
            else:
                for s in retrieved:
                    pairs.append((s, claim))
                    owner.append(i)

        # NLI 배치 처리 (try-except 제거 - 날것 에러 확인용)
        flat_ent, flat_con = [], []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            premises = [p[0][:2000] if p[0] else "" for p in batch]
            claims_list = [p[1][:500] if p[1] else "" for p in batch]

            enc = self.tok(premises, claims_list,
                           return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
            enc.pop("token_type_ids", None)

            # 가드: 범인 입력 탐지
            seq_len = enc["input_ids"].shape[1]
            max_id = enc["input_ids"].max().item()
            min_id = enc["input_ids"].min().item()
            vocab_size = self.nli.config.vocab_size
            max_pos = self.nli.config.max_position_embeddings

            if enc["input_ids"].shape[0] == 0:
                print(f"[GUARD] 빈 배치 발견, batch_idx={i}")
                continue
            if seq_len > max_pos:
                print(f"[GUARD] 길이 초과 {seq_len} > {max_pos}, batch_idx={i}")
            if max_id >= vocab_size:
                print(f"[GUARD] id 초과 {max_id} >= {vocab_size}, batch_idx={i}")
            if min_id < 0:
                print(f"[GUARD] 음수 id 발견 {min_id}, batch_idx={i}")
                print(f"  input_ids sample: {enc['input_ids'][0][:20]}")

            # 디버그: 매 100번째 배치 정보
            if i % 100 == 0:
                print(f"[DEBUG] batch_idx={i}, shape={enc['input_ids'].shape}, max_id={max_id}, min_id={min_id}")

            enc = {k: v.to(self.device) for k, v in enc.items()}
            # position_ids 명시적 제공 (RoBERTa gather 버그 회피)
            seq_len = enc["input_ids"].shape[1]
            enc["position_ids"] = torch.arange(seq_len, device=self.device).unsqueeze(0).expand_as(enc["input_ids"])
            p = torch.softmax(self.nli(**enc).logits, dim=-1)
            flat_ent.extend(p[:, self.entail_idx].tolist())
            flat_con.extend(p[:, self.contra_idx].tolist())

        # 집계: 각 doc별 max entailment, max contradiction
        ent = [0.0] * len(docs); con = [0.0] * len(docs)
        for o, e, c in zip(owner, flat_ent, flat_con):
            ent[o] = max(ent[o], e); con[o] = max(con[o], c)

        # 모순이 강하면 entailment 무효화
        probs = [0.0 if c > contra_th else e for e, c in zip(ent, con)]
        labels = [int(p > entail_th) for p in probs]
        return labels, probs, ent, con


class CascadeVerifier:
    """
    단계별 일괄 처리 캐스케이드.
    각 단계는 '미결(pending) claim 집합'을 받아 일부를 확정하고 나머지를 다음 단계로 넘김.
    판정 상태: 1=supported(통과), 0=unsupported(플래그), None=미결
    """
    def __init__(self, nli_scorer, rule_layer=None, llm_judge=None,
                 entail_th=0.5, contra_th=0.5):
        self.nli = nli_scorer          # RetrievalNLIScorer 인스턴스
        self.rule = rule_layer         # 0단계: 수치·날짜 룰 (없으면 skip)
        self.judge = llm_judge         # 4단계: LLM judge (없으면 미결→플래그 처리)
        self.entail_th = entail_th
        self.contra_th = contra_th

    def verify(self, docs, claims):
        n = len(claims)
        verdict = [None] * n                       # 최종 판정
        meta = [{"stage": None} for _ in range(n)] # 어느 단계에서 결정됐는지 추적

        pending = list(range(n))                   # 미결 claim의 인덱스 집합

        # ---------- 0단계: 수치·날짜 룰 ----------
        if self.rule is not None:
            still = []
            for i in pending:
                r = self.rule.judge(docs[i], claims[i])   # 1 / 0 / None 반환 규약
                if r is None:
                    still.append(i)
                else:
                    verdict[i] = r; meta[i]["stage"] = "rule"
            pending = still

        # ---------- 1단계: 검색 top-k × NLI (window=1) ----------
        pending = self._nli_stage(docs, claims, pending, verdict, meta,
                                   stage="nli_w1", window=1, combine=False)

        # ---------- 2단계: top-k 결합 premise ----------
        pending = self._nli_stage(docs, claims, pending, verdict, meta,
                                   stage="nli_combine", window=1, combine=True)

        # ---------- 3단계: window 확대 재시도 ----------
        pending = self._nli_stage(docs, claims, pending, verdict, meta,
                                   stage="nli_w3", window=3, combine=False)

        # ---------- 4단계: LLM judge ----------
        if self.judge is not None and pending:
            results = self.judge.batch_judge(
                [docs[i] for i in pending], [claims[i] for i in pending])
            for i, r in zip(pending, results):
                verdict[i] = r; meta[i]["stage"] = "judge"
            pending = []

        # 남은 미결은 보수적으로 플래그(unsupported)
        for i in range(n):
            if verdict[i] is None:
                verdict[i] = 0; meta[i]["stage"] = "unresolved_flag"

        return verdict, meta

    def _nli_stage(self, docs, claims, pending, verdict, meta, stage, window, combine):
        """미결 claim에 대해 NLI 1패스. supported(통과)만 확정, 나머지는 미결 유지."""
        if not pending:
            return pending

        sub_docs   = [docs[i] for i in pending]
        sub_claims = [claims[i] for i in pending]

        labels, probs, ent, con = self.nli.score(
            sub_docs, sub_claims, window=window, combine=combine,
            entail_th=self.entail_th, contra_th=self.contra_th)

        still = []
        for idx_in_sub, i in enumerate(pending):
            if con[idx_in_sub] > self.contra_th:
                # 강한 모순 → 즉시 플래그 확정 (다음 단계로 안 넘김)
                verdict[i] = 0; meta[i]["stage"] = stage + "_contra"
            elif labels[idx_in_sub] == 1:
                verdict[i] = 1; meta[i]["stage"] = stage   # supported 통과
            else:
                still.append(i)                            # 미결 → 다음 단계
        return still


if __name__ == "__main__":
    # 기존 코드의 scorer 생성부를 이걸로 교체
    path = r".\klue-roberta-nli"
    emb_path = r".\embedding"
    scorer = RetrievalNLIScorer(path, emb_path)

    # (doc, claim, 예상라벨, 케이스설명)
    test_cases = [
        # --- Doc 1: 기업 실적 (숫자/날짜) ---
        ("에이콤 코퍼레이션은 2025년 3분기 매출 4조 2천억 원을 기록했으며, 이는 전년 동기 대비 "
         "15% 증가한 수치다. 회사는 신규 엔지니어 300명을 채용했고 싱가포르에 새 지사를 열었다.",
         "에이콤의 2025년 3분기 매출은 전년 대비 15% 성장했다.", 1, "paraphrase+숫자 일치"),

        ("에이콤 코퍼레이션은 2025년 3분기 매출 4조 2천억 원을 기록했으며, 이는 전년 동기 대비 "
         "15% 증가한 수치다. 회사는 신규 엔지니어 300명을 채용했고 싱가포르에 새 지사를 열었다.",
         "에이콤의 3분기 매출은 감소했다.", 0, "명백한 모순"),

        ("에이콤 코퍼레이션은 2025년 3분기 매출 4조 2천억 원을 기록했으며, 이는 전년 동기 대비 "
         "15% 증가한 수치다. 회사는 신규 엔지니어 300명을 채용했고 싱가포르에 새 지사를 열었다.",
         "에이콤은 엔지니어 500명을 새로 뽑았다.", 0, "숫자 불일치 (300→500)"),

        ("에이콤 코퍼레이션은 2025년 3분기 매출 4조 2천억 원을 기록했으며, 이는 전년 동기 대비 "
         "15% 증가한 수치다. 회사는 신규 엔지니어 300명을 채용했고 싱가포르에 새 지사를 열었다.",
         "에이콤 대표이사는 자사주 매입 계획을 발표했다.", 0, "근거 없음(neutral)"),

        # --- Doc 2: 의학 (강한 paraphrase) ---
        ("이번 임상시험에는 15개 병원에서 환자 1,200명이 등록되었다. 신약을 투여받은 참가자들은 "
         "위약 그룹 대비 증상이 40% 감소했으나, 8%는 두통 등 경미한 부작용을 겪었다.",
         "해당 약물은 위약보다 증상 완화 효과가 유의미하게 컸다.", 1, "강한 paraphrase"),

        ("이번 임상시험에는 15개 병원에서 환자 1,200명이 등록되었다. 신약을 투여받은 참가자들은 "
         "위약 그룹 대비 증상이 40% 감소했으나, 8%는 두통 등 경미한 부작용을 겪었다.",
         "이 약물은 부작용이 전혀 없었다.", 0, "모순"),

        ("이번 임상시험에는 15개 병원에서 환자 1,200명이 등록되었다. 신약을 투여받은 참가자들은 "
         "위약 그룹 대비 증상이 40% 감소했으나, 8%는 두통 등 경미한 부작용을 겪었다.",
         "임상시험은 15개 병원, 1,200명 규모로 진행되었다.", 1, "복수 사실 결합"),

        # --- Doc 3: 행정/뉴스형 (부분 지지 함정 + 한국어 특유 표현) ---
        ("시의회는 화요일 신규 지하철 노선 건설안을 승인했다. 공사는 2027년 3월 착공 예정이며 "
         "총사업비는 약 2조 원으로 추산된다. 일부 주민들은 공사 소음에 대한 우려를 표명했다.",
         "지하철 노선이 승인됐고 2026년에 착공한다.", 0, "절반만 맞음(연도 틀림)"),

        ("시의회는 화요일 신규 지하철 노선 건설안을 승인했다. 공사는 2027년 3월 착공 예정이며 "
         "총사업비는 약 2조 원으로 추산된다. 일부 주민들은 공사 소음에 대한 우려를 표명했다.",
         "주민들이 공사 소음을 걱정했다.", 1, "paraphrase (표명했다→걱정했다)"),

        ("시의회는 화요일 신규 지하철 노선 건설안을 승인했다. 공사는 2027년 3월 착공 예정이며 "
         "총사업비는 약 2조 원으로 추산된다. 일부 주민들은 공사 소음에 대한 우려를 표명했다.",
         "사업비는 이만억 원 규모다.", 0, "숫자 단위 함정 (2조≠이만억... 사실 같은 값!)", ),

        # --- Doc 4: 금융 도메인 (실전 유사) ---
        ("고객님의 계좌에서 6월 10일 해외 결제 35만 원이 승인되었습니다. 해당 거래에 이상이 있을 경우 "
         "72시간 이내에 이의제기를 접수하셔야 하며, 카드 분실 시 즉시 정지 신청이 가능합니다.",
         "이의제기는 사흘 안에 해야 한다.", 1, "단위 변환 (72시간=사흘)"),

        ("고객님의 계좌에서 6월 10일 해외 결제 35만 원이 승인되었습니다. 해당 거래에 이상이 있을 경우 "
         "72시간 이내에 이의제기를 접수하셔야 하며, 카드 분실 시 즉시 정지 신청이 가능합니다.",
         "해외 결제는 거절되었다.", 0, "모순 (승인↔거절)"),
    ]

    docs    = [t[0] for t in test_cases]
    claims  = [t[1] for t in test_cases]
    golds   = [t[2] for t in test_cases]
    descs   = [t[3] for t in test_cases]

    pred_label, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)

    correct = 0
    for i, (pred, prob, gold, desc) in enumerate(zip(pred_label, raw_prob, golds, descs)):
        ok = "OK " if pred == gold else "FAIL"
        correct += (pred == gold)
        print(f"[{ok}] #{i} pred={pred} gold={gold} prob={prob:.4f}  ({desc})")

    print(f"\n{correct}/{len(golds)} correct")