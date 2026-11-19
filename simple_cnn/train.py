import argparse, json, os, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import get_model

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_transforms(name):
    size = 100 if name == "simple_cnn" else 224
    base = [transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]
    if name == "simple_cnn":
        train_tf = transforms.Compose(base)
    else:
        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return train_tf, transforms.Compose(base)


def load_data(data_dir, name, bs, seed=42):
    train_tf, eval_tf = get_transforms(name)
    full_train = datasets.ImageFolder(data_dir, train_tf)
    full_eval = datasets.ImageFolder(data_dir, eval_tf)
    n = len(full_train)
    n_tr, n_val = int(0.7 * n), int(0.15 * n)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    tr_i = idx[:n_tr]
    va_i = idx[n_tr:n_tr + n_val]
    te_i = idx[n_tr + n_val:]
    return (DataLoader(Subset(full_train, tr_i), batch_size=bs, shuffle=True, num_workers=0),
            DataLoader(Subset(full_eval, va_i), batch_size=bs, shuffle=False, num_workers=0),
            DataLoader(Subset(full_eval, te_i), batch_size=bs, shuffle=False, num_workers=0),
            full_train.classes)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["simple_cnn", "improved_cnn", "shufflenet"])
    ap.add_argument("--data_dir", default="data/Fish_Data/images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dl, val_dl, test_dl, classes = load_data(a.data_dir, a.model, a.bs)
    print("Kelas:", classes, "| jumlah:", len(classes))
    assert len(classes) > 1, "Hanya 1 kelas terbaca, periksa --data_dir!"

    model = get_model(a.model, len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("outputs", exist_ok=True)
    history_data = {"history": [], "train_time_s": 0.0}
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = run_epoch(model, val_dl, criterion, device)
        history_data["history"].append(dict(
            epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))

        print(f"Epoch {ep}/{a.epochs}")
        print("-" * 15)
        print(f"Train Loss: {tl:.4f} | Acc: {ta:.4f}")
        print(f"Test Loss : {vl:.4f} | Acc: {va:.4f}\n")

    history_data["train_time_s"] = time.time() - t0
    print(f"Training selesai dalam {int(history_data['train_time_s'])} detik")

    torch.save({"state": model.state_dict(), "classes": classes},
               f"outputs/{a.model}.pth")
    with open(f"outputs/{a.model}_history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    h = history_data["history"]
    epochs = [e["epoch"] for e in h]
    plt.figure()
    plt.plot(epochs, [e["train_loss"] for e in h], color="#931a00", label="Training")
    plt.plot(epochs, [e["val_loss"] for e in h], color="#3399e6", label="Testing")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"outputs/training_{a.model}.png")
    plt.close()
    print(f"Grafik loss disimpan: outputs/training_{a.model}.png")

    from evaluasi import evaluate
    evaluate(model, test_dl, classes, device, name=a.model)


if __name__ == "__main__":
    main()