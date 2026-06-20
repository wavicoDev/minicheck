"""
A.X-Encoder vs KF-DeBERTa 모델 성능 비교 평가
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
import numpy as np
import time

from financial_classifier import FinancialClassifier, TextClassificationDataset


MODELS = {
    "A.X-Encoder": "skt/A.X-Encoder-base",
    "KF-DeBERTa": "kakaobank/kf-deberta-base"
}


class ModelComparator:
    """Compare multiple models on the same dataset"""

    def __init__(self, device: str = None, max_length: int = 256, seed: int = 42):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.seed = seed
        self.models: Dict[str, nn.Module] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.results: Dict[str, dict] = {}
        self.predictions: Dict[str, List[dict]] = {}  # 각 문장별 예측 결과

        # Data splits
        self.train_texts = []
        self.train_labels = []
        self.val_texts = []
        self.val_labels = []
        self.test_texts = []
        self.test_labels = []

    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """Split data into train/val/test with fixed seed"""
        # First split: train vs (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=self.seed
        )

        # Second split: val vs test
        val_size = val_ratio / (1 - train_ratio)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            train_size=val_size,
            stratify=temp_labels,
            random_state=self.seed
        )

        self.train_texts, self.train_labels = train_texts, train_labels
        self.val_texts, self.val_labels = val_texts, val_labels
        self.test_texts, self.test_labels = test_texts, test_labels

        print(f"Data split (seed={self.seed}):")
        print(f"  Train: {len(self.train_texts)} samples")
        print(f"  Val:   {len(self.val_texts)} samples")
        print(f"  Test:  {len(self.test_texts)} samples")

    def train_model(
        self,
        model_key: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """Train a single model"""
        model_name = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Training: {model_key} ({model_name})")
        print(f"{'='*60}")

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = FinancialClassifier(model_name).to(self.device)

        # Create dataset and dataloader
        train_dataset = TextClassificationDataset(
            self.train_texts, self.train_labels, tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Store trained model and tokenizer
        self.models[model_key] = model
        self.tokenizers[model_key] = tokenizer

    def evaluate_model(self, model_key: str, batch_size: int = 8) -> dict:
        """Evaluate a single model on test set"""
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        model.eval()

        # Create test dataset
        test_dataset = TextClassificationDataset(
            self.test_texts, self.test_labels, tokenizer, self.max_length
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        all_preds = []
        all_labels = []
        all_confidences = []
        total_time = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                start_time = time.time()
                logits = model(input_ids, attention_mask)
                total_time += time.time() - start_time

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confidences = probs.max(dim=1).values.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_confidences.extend(confidences)

        # Store per-sentence predictions
        label_map = {0: "비금융", 1: "금융"}
        self.predictions[model_key] = [
            {
                "text": self.test_texts[i],
                "true_label": label_map[all_labels[i]],
                "pred_label": label_map[all_preds[i]],
                "confidence": all_confidences[i],
                "correct": all_labels[i] == all_preds[i]
            }
            for i in range(len(self.test_texts))
        ]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, pos_label=1),
            "recall": recall_score(all_labels, all_preds, pos_label=1),
            "f1": f1_score(all_labels, all_preds, pos_label=1),
            "avg_confidence": float(np.mean(all_confidences)),
            "min_confidence": float(np.min(all_confidences)),
            "inference_time_ms": (total_time / len(self.test_texts)) * 1000
        }

        self.results[model_key] = metrics
        return metrics

    def compare_all(self, epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """Train and evaluate all models"""
        for model_key in MODELS:
            self.train_model(model_key, epochs, batch_size, learning_rate)
            self.evaluate_model(model_key, batch_size)

    def print_results(self):
        """Print comparison results as a table"""
        print(f"\n{'='*80}")
        print(f"{'모델 성능 비교 결과':^80}")
        print(f"{'='*80}")
        print(f"{'Model':<15} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'Time(ms)':>8}")
        print(f"{'-'*15}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

        for model_key, metrics in self.results.items():
            print(
                f"{model_key:<15} | "
                f"{metrics['accuracy']:>8.4f} | "
                f"{metrics['precision']:>9.4f} | "
                f"{metrics['recall']:>6.4f} | "
                f"{metrics['f1']:>6.4f} | "
                f"{metrics['inference_time_ms']:>8.2f}"
            )

        print(f"{'='*80}")

        # Winner summary
        best_acc = max(self.results.items(), key=lambda x: x[1]["accuracy"])
        best_f1 = max(self.results.items(), key=lambda x: x[1]["f1"])
        fastest = min(self.results.items(), key=lambda x: x[1]["inference_time_ms"])

        print(f"\nBest Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
        print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
        print(f"Fastest:       {fastest[0]} ({fastest[1]['inference_time_ms']:.2f} ms/sample)")

        # Print detailed predictions
        self.print_detailed_predictions()

    def print_detailed_predictions(self):
        """Print per-sentence predictions for all models"""
        print(f"\n{'='*100}")
        print(f"{'문장별 예측 결과 비교':^100}")
        print(f"{'='*100}")

        for i, text in enumerate(self.test_texts):
            print(f"\n[{i+1}] {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"    정답: {self.predictions[list(MODELS.keys())[0]][i]['true_label']}")
            print(f"    {'-'*50}")
            for model_key in MODELS:
                pred = self.predictions[model_key][i]
                status = "O" if pred["correct"] else "X"
                print(f"    {model_key:<12}: {pred['pred_label']} (신뢰도: {pred['confidence']:.2%}) [{status}]")


# 샘플 데이터
SAMPLE_DATA = {
    "texts": [
        # 금융 (label=1)
        "주식 시장이 급등하면서 코스피 지수가 사상 최고치를 경신했습니다.",
        "금리 인상으로 인해 대출 이자 부담이 늘어날 전망입니다.",
        "삼성전자 주가가 어제 대비 5% 상승했습니다.",
        "비트코인 가격이 5만 달러를 돌파했습니다.",
        "연준의 금리 결정이 글로벌 증시에 영향을 미쳤습니다.",
        "펀드 수익률이 작년 대비 15% 상승했습니다.",
        "환율이 달러당 1300원을 넘어섰습니다.",
        "부동산 담보대출 금리가 연 4%대로 올랐습니다.",
        "코스닥 시장에서 바이오 관련주가 강세를 보이고 있습니다.",
        "신용카드 연체율이 증가하고 있어 금융권이 주의를 기울이고 있습니다.",
        "국채 금리가 하락하면서 채권 가격이 상승했습니다.",
        "원/달러 환율이 급락하여 수출기업 실적에 영향을 줄 전망입니다.",
        "카카오뱅크의 신규 적금 상품이 출시되었습니다.",
        "증권사들이 올해 목표 주가를 상향 조정했습니다.",
        "연금저축 가입자 수가 역대 최고치를 기록했습니다.",
        "미국 연방준비제도가 기준금리를 동결했습니다.",
        # 비금융 (label=0)
        "오늘 서울 날씨는 맑고 기온은 25도입니다.",
        "새로운 영화가 개봉하여 많은 관객을 모았습니다.",
        "건강을 위해 매일 운동하는 것이 좋습니다.",
        "맛있는 레시피를 공유합니다.",
        "여행 가기 좋은 계절이 왔습니다.",
        "새로운 스마트폰이 출시되었습니다.",
        "오늘 점심 메뉴는 김치찌개입니다.",
        "주말에 가족과 함께 캠핑을 다녀왔습니다.",
        "주말에 친구들과 영화를 보러 갔습니다.",
        "오늘 저녁에 맛있는 파스타를 만들 예정입니다.",
        "새로운 드라마가 시청률 1위를 기록했습니다.",
        "올해 여름 휴가로 제주도 여행을 계획하고 있습니다.",
        "카페에서 책을 읽으며 여유로운 시간을 보냈습니다.",
        "아이들과 함께 공원에서 자전거를 탔습니다.",
        "최근 인기 있는 맛집을 방문했습니다.",
        "주말에 등산을 가서 산 정상에 올랐습니다.",
    ],
    "labels": [1]*16 + [0]*16  # 16 financial + 16 non-financial
}


if __name__ == "__main__":
    print("=" * 80)
    print("A.X-Encoder vs KF-DeBERTa 금융/비금융 분류 성능 비교")
    print("=" * 80)

    # Initialize comparator
    comparator = ModelComparator(seed=42)

    # Prepare data
    print("\n[1] 데이터 준비")
    comparator.prepare_data(
        texts=SAMPLE_DATA["texts"],
        labels=SAMPLE_DATA["labels"],
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Train and evaluate all models
    print("\n[2] 모델 학습 및 평가")
    comparator.compare_all(
        epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )

    # Print comparison results
    print("\n[3] 결과 비교")
    comparator.print_results()
