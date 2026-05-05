import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class SVM:
    """
    Soft-margin Support Vector Machine (Primal form, Linear kernel).

    Tối ưu hàm mục tiêu hinge loss bằng Stochastic Gradient Descent (SGD):

        L(w, b) = 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i * f(x_i)))

    trong đó f(x) = w^T * x + b là hàm quyết định tuyến tính.

    Gradient của L theo từng sample (x_i, y_i):
        - Nếu y_i * f(x_i) >= 1 (không vi phạm margin):
            dL/dw = w,  dL/db = 0
        - Nếu y_i * f(x_i) < 1 (vi phạm margin):
            dL/dw = w + C * (-y_i * x_i),  dL/db = C * (-y_i)

    Quy ước nhãn:
        - NORMAL    : y = -1
        - PNEUMONIA : y =  1

    Attributes:
        C (float): Hệ số penalty cho các điểm vi phạm margin.
                   C lớn → margin cứng hơn, C nhỏ → margin mềm hơn.
        lr (float): Learning rate cho SGD.
        n_iterations (int): Số epoch training.
        W (np.ndarray): Trọng số, shape (dim,).
        b (float): Bias.
        losses (list): Lịch sử loss theo từng sample.
    """

    def __init__(self, C: float = 1.0, lr: float = 1e-5, n_iterations: int = 100):
        self.C = C
        self.lr = lr
        self.n_iterations = n_iterations
        self.W = None
        self.b = 0
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train SVM bằng SGD trên từng sample.

        Args:
            X (np.ndarray): Training data, shape (N, dim).
            y (np.ndarray): Nhãn, shape (N,), giá trị -1 hoặc 1.
        """
        N, dim = X.shape

        # Khởi tạo tham số
        self.W = np.zeros((dim,))
        self.b = 0
        self.losses = []

        for epoch in range(self.n_iterations):
            pbar = tqdm(range(N), desc=f"Epoch {epoch} - Training")
            for ith in pbar:
                x = X[ith]   # vector ảnh, shape (dim,)
                a = y[ith]   # nhãn, -1 hoặc 1

                # Tính prediction và loss cho sample hiện tại
                y_pred = self.predict(np.expand_dims(x, axis=0))  # shape (1,)
                loss = self.loss_fn(a, y_pred)
                self.losses.append(loss)

                pbar.set_postfix({"Loss": loss})

                # Tính gradient theo hinge loss
                if a * y_pred >= 1:
                    # Sample nằm đúng phía và ngoài margin
                    # Chỉ có gradient từ regularization term
                    dW = self.W
                    db = 0
                else:
                    # Sample vi phạm margin (hoặc sai phía)
                    # Gradient từ cả regularization lẫn hinge loss
                    dW = self.W + self.C * (-a * x)
                    db = self.C * (-a)

                # Cập nhật tham số theo hướng gradient âm (gradient descent)
                self.W = self.W - self.lr * dW
                self.b = self.b - self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tính raw score của hàm quyết định f(x) = w^T * x + b.

        Args:
            X (np.ndarray): Input data, shape (N, dim).

        Returns:
            scores (np.ndarray): Raw scores, shape (N,).
                                 score > 0 → dự đoán PNEUMONIA (1)
                                 score < 0 → dự đoán NORMAL (-1)
        """
        return X @ self.W + self.b

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Tính hinge loss kết hợp với L2 regularization:

            L = 0.5 * ||W||^2 + C * sum(max(0, 1 - y * y_hat))

        Args:
            y: Nhãn thực, scalar hoặc array.
            y_hat: Predicted scores, scalar hoặc array.

        Returns:
            loss (float): Giá trị loss.
        """
        regularization = 0.5 * np.dot(self.W.T, self.W)
        hinge = self.C * np.where(1 - y * y_hat < 0, 0, 1 - y * y_hat).sum()
        return regularization + hinge

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Đánh giá model trên tập dữ liệu bằng Precision, Recall, F1.

        Args:
            X (np.ndarray): Input data, shape (N, dim).
            y (np.ndarray): Nhãn thực, shape (N,), giá trị -1 hoặc 1.

        Returns:
            dict: {
                "Precision": float,
                "Recall"   : float,
                "F1"       : float
            }
        """
        # Lấy raw score rồi threshold tại 0
        y_pred = self.predict(X)
        y_pred = np.where(y_pred >= 0, 1, -1)

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
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Khởi tạo và train model
    model = SVM(
        C=1,
        lr=1e-5,
        n_iterations=100
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    metrics = model.get_metrics(X_test, y_test)
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1        : {metrics['F1']:.4f}")
