import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm
import sacrebleu
from sacrebleu.metrics import BLEU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "mengzi-t5-base"
SAVE_DIR = "./qa_model"
BEST_DIR = "./qa_model_best"


tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_text = f"question: {item['question']}  context: {item['context']}"
        target_text = item["answer"]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


def load_model():
    if os.path.exists(SAVE_DIR):
        print("加载已有 ckpt...")
        model = T5ForConditionalGeneration.from_pretrained(SAVE_DIR)
    else:
        print("从预训练模型初始化...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    return model.to(device)


def train():

    model = load_model()

    train_dataset = QADataset("DuReaderQG/train.json", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_epochs = 5
    loss_history = []

    best_loss = float("inf")

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        loop = tqdm(train_loader)

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} Loss: {avg_loss}")

        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(BEST_DIR)
            tokenizer.save_pretrained(BEST_DIR)
            print("保存最佳模型")

    # 绘图
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()


def evaluate():

    model = T5ForConditionalGeneration.from_pretrained(BEST_DIR).to(device)
    model.eval()

    dev_dataset = QADataset("DuReaderQG/dev.json", tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    refs = []
    hyps = []

    with torch.no_grad():
        for batch in tqdm(dev_loader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4
            )

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = torch.where(
                labels != -100,
                labels,
                tokenizer.pad_token_id
            )

            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            hyps.extend(preds)
            refs.extend([t for t in targets])   # 注意这里改了

    # sacrebleu 新API
    bleu = BLEU(smooth_method='exp')
    score = bleu.corpus_score(hyps, [refs])

    print("BLEU:", score.score)
    print("BLEU-1:", score.precisions[0])
    print("BLEU-2:", score.precisions[1])
    print("BLEU-3:", score.precisions[2])
    print("BLEU-4:", score.precisions[3])

    return score.precisions[0], score.precisions[1], score.precisions[2], score.precisions[3]


def predict(context, question):

    model = T5ForConditionalGeneration.from_pretrained(BEST_DIR).to(device)
    model.eval()

    input_text = f"question: {question}  context: {context}"

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


if __name__ == "__main__":

    train()
    bleu1, bleu2, bleu3, bleu4 = evaluate()
    context = "违规分为..."
    question = "淘宝扣分什么时候清零"
    print("预测结果：", predict(context, question))