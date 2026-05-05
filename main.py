import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

from svm import SVM
from svm_sklearn import SVMSklearn

# Tự động cd vào folder chứa main.py khi chạy từ VS Code
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = "data/chest_xray"


def collect_data(split: str = "train"):
    """
    Nạp và tiền xử lý ảnh X-Ray từ dataset Chest X-Ray Images (Pneumonia).

    """
    images = []
    labels = []

    if split == "train":
        for img_file in os.listdir(os.path.join(BASE_DIR, split, "NORMAL")):
            img = cv.imread(os.path.join(BASE_DIR, split, "NORMAL", img_file))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img.reshape(-1) / 255.0)
            labels.append(-1)

        for img_file in os.listdir(os.path.join(BASE_DIR, split, "BACTERIAL")):
            img = cv.imread(os.path.join(BASE_DIR, split, "BACTERIAL", img_file))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img.reshape(-1) / 255.0)
            labels.append(1)

        for img_file in os.listdir(os.path.join(BASE_DIR, split, "VIRAL")):
            img = cv.imread(os.path.join(BASE_DIR, split, "VIRAL", img_file))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img.reshape(-1) / 255.0)
            labels.append(1)

    elif split == "test":
        for img_file in os.listdir(os.path.join(BASE_DIR, split)):
            img = cv.imread(os.path.join(BASE_DIR, split, img_file))
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img.reshape(-1) / 255.0)
            name = img_file.lower()
            labels.append(-1 if "_normal_" in name else 1)

    else:
        raise ValueError(f"split phải là 'train' hoặc 'test', nhận được: '{split}'")

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def plot_loss_curve(losses, n_iterations, n_train):
    losses = np.array(losses)
    epoch_losses = [losses[e * n_train:(e + 1) * n_train].mean()
                    for e in range(n_iterations)]

    plt.figure(figsize=(10, 5))
    plt.plot(range(n_iterations), epoch_losses, linewidth=2, color="steelblue")
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Average Loss", fontsize=13)
    plt.title("Training Loss Curve - Soft-margin SVM (SGD)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=100)
    plt.show()
    print(f"  Final loss: {epoch_losses[-1]:.4f}")


def plot_comparison(results):
    models = list(results.keys())
    metrics_names = ["Precision", "Recall", "F1"]
    x = np.arange(len(models))
    width = 0.25
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        values = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric,
                      color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("So sánh kết quả các mô hình SVM", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=100)
    plt.show()


def print_results(results, title="Kết quả trên tập Test"):
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")
    print(f"{'Model':<22} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"{'=' * 52}")
    for name, m in results.items():
        print(f"{name:<22} {m['Precision']:>9.4f} {m['Recall']:>9.4f} {m['F1']:>9.4f}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    np.random.seed(42)

    # Load data
    print("=" * 60)
    print(f"{'GIẢI ĐOẠN: NẠP DỮ LIỆU':^60}")
    print("=" * 60)
    X_train, y_train = collect_data("train")
    X_test, y_test = collect_data("test")
    print(f"  Train set: {X_train.shape} | Test set: {X_test.shape}")

    # Danh sách lưu kết quả để so sánh
    train_results = {}
    test_results = {}

    # 1. Assignment 1: SVM NumPy + SGD
    print("\n" + "=" * 60)
    print(f"{'ASSIGNMENT 1: SVM NUMPY + SGD':^60}")
    print("=" * 60)
    svm_numpy = SVM(C=1, lr=1e-5, n_iterations=100)
    svm_numpy.fit(X_train, y_train)
    
    # Đánh giá trên cả 2 tập để kiểm tra Overfitting
    train_results["SVM NumPy (SGD)"] = svm_numpy.get_metrics(X_train, y_train)
    test_results["SVM NumPy (SGD)"] = svm_numpy.get_metrics(X_test, y_test)

    # 2. Assignment 2: SVM sklearn (SMO)
    print("\n" + "=" * 60)
    print(f"{'ASSIGNMENT 2: SVM SKLEARN (SMO)':^60}")
    print("=" * 60)
    
    # Linear Kernel
    svm_linear = SVMSklearn(kernel="linear", C=1.0)
    svm_linear.fit(X_train, y_train)
    train_results["sklearn (Linear)"] = svm_linear.get_metrics(X_train, y_train)
    test_results["sklearn (Linear)"] = svm_linear.get_metrics(X_test, y_test)

    # RBF Kernel + PCA
    svm_rbf = SVMSklearn(kernel="rbf", C=1.0, n_components=150)
    svm_rbf.fit(X_train, y_train)
    train_results["sklearn (RBF+PCA)"] = svm_rbf.get_metrics(X_train, y_train)
    test_results["sklearn (RBF+PCA)"] = svm_rbf.get_metrics(X_test, y_test)

    # --- IN BẢNG SO SÁNH ĐỂ KIỂM TRA OVERFITTING ---
    print("\n" + " " * 15 + "BẢNG KIỂM TRA OVERFITTING (F1-SCORE)")
    print("-" * 60)
    print(f"{'Model':<22} | {'Train F1':>15} | {'Test F1':>15}")
    print("-" * 60)
    for name in train_results.keys():
        f1_train = train_results[name]['F1']
        f1_test = test_results[name]['F1']
        print(f"{name:<22} | {f1_train:>15.4f} | {f1_test:>15.4f}")
    print("-" * 60)

    # In kết quả chi tiết trên tập Test và vẽ biểu đồ
    print_results(test_results, title="Kết quả chi tiết trên tập Test")
    plot_comparison(test_results)
    plot_loss_curve(svm_numpy.losses, svm_numpy.n_iterations, len(X_train))
