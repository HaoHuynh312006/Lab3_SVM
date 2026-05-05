import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score


class SVMSklearn:
    """
    Wrapper class cho sklearn SVC, phục vụ Assignment 2.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0, n_components: int = None):
        self.kernel = kernel
        self.C = C
        self.n_components = n_components
        self.model = SVC(kernel=kernel, C=C)

        # Chỉ dùng PCA khi kernel='rbf' và n_components được chỉ định
        self.pca = PCA(n_components=n_components) if n_components else None

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Áp dụng PCA nếu có.
        """
        if self.pca is None:
            return X
        if fit:
            return self.pca.fit_transform(X)
        return self.pca.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train SVM bằng sklearn (dùng SMO bên trong).
        """
        print(f"Training SVM (kernel='{self.kernel}', C={self.C}"
              + (f", PCA={self.n_components}" if self.pca else "") + ")...")

        X_transformed = self._transform(X, fit=True)
        self.model.fit(X_transformed, y)
        print("Done!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn cho dữ liệu đầu vào.
        """
        X_transformed = self._transform(X, fit=False)
        return self.model.predict(X_transformed)

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Đánh giá model bằng Precision, Recall, F1.
        """
        y_pred = self.predict(X)

        P  = precision_score(y, y_pred, pos_label=1)
        R  = recall_score(y, y_pred, pos_label=1)
        f1 = f1_score(y, y_pred, pos_label=1)

        return {
            "Precision": P,
            "Recall"   : R,
            "F1"       : f1
        }


if __name__ == "__main__":
    from main import collect_data

    np.random.seed(42)

    print("Loading data...")
    X_train, y_train = collect_data("train")
    X_test, y_test   = collect_data("test")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    results = {}

    # --- Linear kernel (không cần PCA) ---
    svm_linear = SVMSklearn(kernel="linear", C=1.0)
    svm_linear.fit(X_train, y_train)
    results["sklearn (Linear)"] = svm_linear.get_metrics(X_test, y_test)

    # --- RBF kernel (dùng PCA 150 components) ---
    svm_rbf = SVMSklearn(kernel="rbf", C=1.0, n_components=150)
    svm_rbf.fit(X_train, y_train)
    results["sklearn (RBF)"] = svm_rbf.get_metrics(X_test, y_test)

    # In kết quả
    print("\n" + "=" * 52)
    print(f"{'Model':<22} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print("=" * 52)
    for name, m in results.items():
        print(f"{name:<22} {m['Precision']:>9.4f} {m['Recall']:>9.4f} {m['F1']:>9.4f}")
    print("=" * 52)
