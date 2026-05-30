import os

import torch
import torch.nn as nn
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


def run_epoch(
    model,
    loader,
    optimizer=None,
    device=None,
    class_weights: torch.Tensor | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    loss_fn = None
    if is_train and class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    batch_idx = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        amp_enabled = use_amp and device is not None and device.type == "cuda"
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if loss_fn is not None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = loss_fn(outputs.logits, labels)
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

        if is_train:
            scaled_loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_idx += 1
        if is_train and batch_idx % log_every == 0:
            print(f"  train batch {batch_idx}/{n_batches} loss={loss.item():.4f}")

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if is_train and batch_idx % grad_accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    device = torch.device(config.TORCH_DEVICE)
    print(f"--- Step 1: Initialize Tokenizer and Model (device={device}) ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.XLMR_MODEL_NAME,
        num_labels=5,
    )
    model.to(device)
    if config.XLMR_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(config.XLMR_MODEL_NAME)

    df = utils.load_and_standardize_data(config.RAW_DATA_PATH)
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )

    train_labels = df_train["user_rating"].values - 1
    val_labels = df_val["user_rating"].values - 1
    train_ratings = df_train["user_rating"].values

    class_weight_np = None
    class_weights_tensor = None
    if config.XLMR_USE_CLASS_WEIGHT:
        class_weight_np = utils.compute_class_weights(
            train_ratings,
            low_star_boost=config.XLMR_LOW_STAR_BOOST,
        )
        class_weights_tensor = torch.tensor(
            class_weight_np, dtype=torch.float32, device=device
        )
    utils.print_rating_distribution(
        train_ratings, class_weight_np, label="train"
    )

    batch_size = config.BATCH_SIZE
    grad_accum_steps = max(1, config.XLMR_GRAD_ACCUM_STEPS)
    max_length = config.MAX_LENGTH

    train_dataset = ReviewDataset(
        df_train["text"].values,
        train_labels,
        tokenizer,
        max_length,
    )
    val_dataset = ReviewDataset(
        df_val["text"].values,
        val_labels,
        tokenizer,
        max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = None
    if config.XLMR_USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.XLMR_LR_SCHEDULER_FACTOR,
            patience=config.XLMR_LR_SCHEDULER_PATIENCE,
        )

    train_class_weights = class_weights_tensor if config.XLMR_USE_CLASS_WEIGHT else None
    use_amp = config.XLMR_USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    effective_batch = batch_size * grad_accum_steps
    print(
        f"--- Step 2: Training (batch={batch_size}, "
        f"accum={grad_accum_steps}, effective_batch={effective_batch}, "
        f"max_length={max_length}, amp={use_amp}) ---"
    )

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=train_class_weights,
            scaler=scaler if use_amp else None,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            device=device,
            class_weights=None,
            use_amp=use_amp,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  lr={current_lr:.2e}")

        if patience_counter >= config.XLMR_EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val_acc improvement for "
                f"{config.XLMR_EARLY_STOPPING_PATIENCE} epochs)"
            )
            break

    print("--- Step 3: Saving Weights Manually ---")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch} (val_acc={best_val_acc:.4f})")
    else:
        print("Warning: no improvement seen; saving final epoch weights.")

    out_dir = config.XLMR_ARTIFACTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Done PyTorch loop training! Saved to {out_dir}")


if __name__ == "__main__":
    main()
