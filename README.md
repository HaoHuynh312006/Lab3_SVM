# Lab3_SVM
Chest X-Ray Pneumonia Classification using SVMDự án này triển khai mô hình Support Vector Machine (SVM) để phân loại ảnh X-Quang ngực nhằm chẩn đoán viêm phổi (Pneumonia). Bài làm bao gồm hai phần chính: tự cài đặt SVM bằng NumPy (Assignment 1) và sử dụng thư viện Scikit-learn (Assignment 2).  
1. Thành viên thực hiện
   Họ tên: Huỳnh Gia HàoMSSV: 24520457  
   Lớp: Khoa Khoa học dữ liệu - UIT
2. Cấu trúc thư mục
Để chạy được mã nguồn, vui lòng tổ chức thư mục dữ liệu như sau (không push thư mục data lên repo):
.
├── assets/                 # Chứa các biểu đồ kết quả (png)
├── data/
│   └── chest_xray/
│       ├── train/          # Thư mục chứa 5216 ảnh train
│       └── test/           # Thư mục chứa 40 ảnh test
├── main_2.py               # Script thực thi chính
├── svm_2.py                # Implement SVM từ NumPy (SGD)
├── svm_sklearn.py          # Implement SVM từ sklearn (SMO)
├── requirements.txt        # Các thư viện cần thiết
└── README.md
3. Cài đặt và sử dụngCài đặt các thư viện cần thiết:
   Bash pip install -r requirements.txt
Chạy chương trình để huấn luyện và đánh giá:
   Bash python main.py
4. Phương pháp triển khai
   Assignment 1: NumPy & SGD
   Mô hình: Soft-margin SVM (Primal form).
   Tối ưu hóa: Stochastic Gradient Descent (SGD) với hàm lỗi Hinge Loss.
   Ưu điểm: Hiểu rõ bản chất cập nhật trọng số w và b qua từng vòng lặp.
   Assignment 2: Scikit-learn & SMO
   Mô hình: SVC (C-Support Vector Classification) sử dụng thuật toán SMO giải bài toán đối ngẫu.
   Kỹ thuật bổ trợ:
     Linear Kernel: Để so sánh trực tiếp với Assignment 1.
     RBF Kernel + PCA: Sử dụng PCA giảm xuống 150 chiều để xử lý dữ liệu ảnh 128 x 128 nhằm tăng tốc độ huấn luyện cho nhân phi tuyến.
5. Kết quả thực nghiệm
   Bảng so sánh kết quả trên tập Test
    Model,Kernel,Precision,Recall,F1-Score
    SVM NumPy (SGD),Linear,0.7838,0.9667,0.8657
    sklearn (Linear),Linear,0.7692,1.0000,0.8696
    sklearn (RBF+PCA),RBF,0.9091,1.0000,0.9524
   <img width="1000" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/735fbadc-5c0e-4fbe-b141-3046333a719c" />
   <img width="1100" height="600" alt="comparison" src="https://github.com/user-attachments/assets/2b2a1359-e3d3-4bed-a36e-cde59f90b5d9" />
6. Phân tích và Nhận xét
   Độ chính xác: Mô hình sklearn (RBF+PCA) đạt kết quả tốt nhất (F1 ~ 0.95$), cho thấy dữ liệu ảnh y tế có tính phi tuyến cao và Kernel RBF đã xử lý rất hiệu quả.
   Khả năng quát hóa: Chỉ số F1 của mô hình tự cài đặt bằng NumPy tương đương với sklearn (Linear), chứng minh thuật toán SGD đã hội tụ đúng điểm tối ưu.
   Vấn đề Overfitting: Mặc dù Recall đạt 1.0, nhưng qua việc kiểm tra chỉ số trên cả tập Train và Test, mô hình cho thấy sự ổn định và không có dấu hiệu "học tủ" (overfitting) nghiêm trọng, dù tập test còn hạn chế về số lượng.
   <img width="433" height="150" alt="image" src="https://github.com/user-attachments/assets/7e96bf00-d886-4a70-ad7f-8bb02fd75f35" />
