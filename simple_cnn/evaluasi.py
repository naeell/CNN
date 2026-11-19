import argparse, os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)


def evaluate(model, test_dl, classes, device, name="model", save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            targets.extend(y.numpy())

    # ---------- Metrik ringkas (seperti tabel slide 18) ----------
    acc = accuracy_score(targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0)
    print("\nMenjalankan evaluasi...")
    print("\n--- HASIL EVALUASI MODEL PADA DATA TEST ---")
    print(f"Model     : {name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # ---------- Classification report ----------
    report = classification_report(targets, preds, target_names=classes,
                                   digits=4, zero_division=0)
    print("\nCLASSIFICATION REPORT")
    print(report)
    with open(f"{save_dir}/classification_report_{name}.txt", "w") as f:
        f.write(report)
    pd.DataFrame([dict(model=name, accuracy=acc, precision=p, recall=r, f1=f1)]
                 ).to_csv(f"{save_dir}/metrics_{name}.csv", index=False)

    # ---------- Confusion matrix ----------
    cm = confusion_matrix(targets, preds)
    print("CONFUSION MATRIX (jumlah)")
    print(cm)
    cm_norm = cm / cm.sum(axis=1)[:, None]          # seperti slide 16
    df_cm = pd.DataFrame(cm_norm, index=classes, columns=classes)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax[0])
    ax[0].set_title("Confusion Matrix (Jumlah)")
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", ax=ax[1])
    ax[1].set_title("Confusion Matrix (Normalisasi per kelas)")
    for a in ax:
        a.set_xlabel("Predicted Label")
        a.set_ylabel("True Label")
    plt.suptitle(f"Evaluasi {name}")
    plt.tight_layout()
    path = f"{save_dir}/confusion_matrix_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nTersimpan: {path}")
    return acc, p, r, f1


if __name__ == "__main__":
    from train import load_data
    from model import get_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn",
                    choices=["simple_cnn", "improved_cnn", "shufflenet"])
    ap.add_argument("--data_dir", default="data/Fish_Data/images")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_dl, classes = load_data(a.data_dir, a.model, a.bs)
    model = get_model(a.model, len(classes)).to(device)
    ckpt = torch.load(f"outputs/{a.model}.pth", map_location=device)
    model.load_state_dict(ckpt["state"])
    evaluate(model, test_dl, classes, device, name=a.model)