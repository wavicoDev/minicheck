"""
A.X-Encoder 기반 금융/비금융 텍스트 분류기
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Tuple, Optional
import numpy as np


class FinancialClassifier(nn.Module):
    """A.X-Encoder + Classification Head for financial/non-financial classification"""

    def __init__(self, model_name: str = "skt/A.X-Encoder-base", num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class TextClassificationDataset(Dataset):
    """Dataset for text classification"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class FinancialClassifierTrainer:
    """Trainer for FinancialClassifier"""

    LABEL_MAP = {0: "비금융", 1: "금융"}

    def __init__(
        self,
        model_name: str = "skt/A.X-Encoder-base",
        device: Optional[str] = None,
        max_length: int = 256
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FinancialClassifier(model_name).to(self.device)

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1
    ) -> dict:
        """Train the classifier"""
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps
        )
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # Validation
            if val_texts and val_labels:
                val_loss, val_acc = self._evaluate(val_texts, val_labels, batch_size, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return history

    def _evaluate(self, texts: List[str], labels: List[int], batch_size: int, criterion) -> Tuple[float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        dataset = TextClassificationDataset(texts, labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

        return total_loss / len(loader), correct / total

    def predict(self, texts: List[str], return_probs: bool = False) -> List[dict]:
        """Predict financial/non-financial for given texts"""
        self.model.eval()
        results = []

        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))

                result = {
                    "text": text,
                    "prediction": pred,
                    "label": self.LABEL_MAP[pred],
                    "confidence": float(probs[pred])
                }
                if return_probs:
                    result["probs"] = {"비금융": float(probs[0]), "금융": float(probs[1])}
                results.append(result)

        return results

    def save(self, path: str):
        """Save model and tokenizer"""
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """Load model from path"""
        self.model.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# 샘플 데이터 및 테스트
if __name__ == "__main__":
    # 샘플 학습 데이터
    train_texts = [
        # 금융 (label=1)
        "주식 시장이 급등하면서 코스피 지수가 사상 최고치를 경신했습니다.",
        "금리 인상으로 인해 대출 이자 부담이 늘어날 전망입니다.",
        "삼성전자 주가가 어제 대비 5% 상승했습니다.",
        "비트코인 가격이 5만 달러를 돌파했습니다.",
        "연준의 금리 결정이 글로벌 증시에 영향을 미쳤습니다.",
        "펀드 수익률이 작년 대비 15% 상승했습니다.",
        "환율이 달러당 1300원을 넘어섰습니다.",
        "부동산 담보대출 금리가 연 4%대로 올랐습니다.",
        # 비금융 (label=0)
        "오늘 서울 날씨는 맑고 기온은 25도입니다.",
        "새로운 영화가 개봉하여 많은 관객을 모았습니다.",
        "건강을 위해 매일 운동하는 것이 좋습니다.",
        "맛있는 레시피를 공유합니다.",
        "여행 가기 좋은 계절이 왔습니다.",
        "새로운 스마트폰이 출시되었습니다.",
        "오늘 점심 메뉴는 김치찌개입니다.",
        "주말에 가족과 함께 캠핑을 다녀왔습니다.",
    ]
    train_labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    # 테스트 데이터
    test_texts = [
        "코스닥 시장에서 바이오 관련주가 강세를 보이고 있습니다.",
        "주말에 친구들과 영화를 보러 갔습니다.",
        "신용카드 연체율이 증가하고 있어 금융권이 주의를 기울이고 있습니다.",
        "오늘 저녁에 맛있는 파스타를 만들 예정입니다.",
    ]

    print("=" * 60)
    print("A.X-Encoder 기반 금융/비금융 분류기")
    print("=" * 60)

    # 분류기 초기화
    print("\n[1] 모델 로드 중...")
    trainer = FinancialClassifierTrainer(model_name="skt/A.X-Encoder-base")
    print(f"    Device: {trainer.device}")

    # 학습
    print("\n[2] 모델 학습 중...")
    history = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        epochs=5,
        batch_size=4,
        learning_rate=2e-5
    )

    # 추론
    print("\n[3] 테스트 추론 결과:")
    print("-" * 60)
    results = trainer.predict(test_texts, return_probs=True)
    for r in results:
        print(f"텍스트: {r['text'][:50]}...")
        print(f"예측: {r['label']} (신뢰도: {r['confidence']:.2%})")
        print(f"확률: 비금융={r['probs']['비금융']:.2%}, 금융={r['probs']['금융']:.2%}")
        print("-" * 60)
