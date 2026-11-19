import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, test_dl, classes, device, name="model", save_dir="outputs"):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            targets.extend(y.numpy())

    # ---------- 1. CLASSIFICATION REPORT ----------
    print("\nMenjalankan evaluasi...")
    print("\n" + "=" * 60)
    print(f"EVALUASI MODEL: {name}")
    print("=" * 60)
    print(f"Accuracy total: {accuracy_score(targets, preds):.4f}\n")

    report = classification_report(targets, preds, target_names=classes,
                                   digits=4, zero_division=0)
    print("CLASSIFICATION REPORT")
    print(report)
    with open(f"{save_dir}/classification_report_{name}.txt", "w") as f:
        f.write(report)

    # ---------- 2. CONFUSION MATRIX ----------
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax[0])
    ax[0].set_title("Confusion Matrix (Jumlah)")
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax[1])
    ax[1].set_title("Confusion Matrix (% per kelas asli)")
    ax[1].set_xlabel("Predicted Label")
    ax[1].set_ylabel("True Label")

    plt.suptitle(f"Evaluasi {name}")
    plt.tight_layout()
    path = f"{save_dir}/confusion_matrix_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gambar disimpan: {path}")
    print(f"Report disimpan: {save_dir}/classification_report_{name}.txt")

    return accuracy_score(targets, preds), targets, preds


if __name__ == "__main__":
    from train import load_data
    from model import get_model

    name = "simple_cnn"              # ganti: improved_cnn / shufflenet
    data_dir = "data/Fish_Data/images"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_dl, classes = load_data(data_dir, name, 32)
    model = get_model(name, len(classes)).to(device)
    ckpt = torch.load(f"outputs/{name}.pth", map_location=device)
    model.load_state_dict(ckpt["state"])
    evaluate(model, test_dl, classes, device, name=name)