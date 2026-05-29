import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import utils


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def run_epoch(model, loader, optimizer=None, device=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    device = torch.device(config.TORCH_DEVICE)
    print(f"--- Step 1: Initialize Tokenizer and Model (device={device}) ---")
    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.XLMR_MODEL_NAME,
        num_labels=5,
    )
    model.to(device)

    df = utils.load_and_standardize_data(config.RAW_DATA_PATH)
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )

    train_labels = df_train["user_rating"].values - 1
    val_labels = df_val["user_rating"].values - 1

    train_dataset = ReviewDataset(
        df_train["text"].values,
        train_labels,
        tokenizer,
        config.MAX_LENGTH,
    )
    val_dataset = ReviewDataset(
        df_val["text"].values,
        val_labels,
        tokenizer,
        config.MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    print("--- Step 2: Training via PyTorch Custom Loop (No HF Trainer) ---")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = run_epoch(model, val_loader, device=device)
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    print("--- Step 3: Saving Weights Manually ---")
    out_dir = os.path.join(config.ARTIFACTS_DIR, "xlmr_model")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Done PyTorch loop training! Saved to {out_dir}")


if __name__ == "__main__":
    main()
