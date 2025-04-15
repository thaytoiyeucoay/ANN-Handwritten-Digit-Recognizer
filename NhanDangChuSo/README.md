# Nhận dạng chữ số viết tay sử dụng ANN

Dự án này sử dụng mạng nơ-ron nhân tạo (ANN) để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST. Mô hình được xây dựng để phân loại 10 lớp, tương ứng với các chữ số từ 0 đến 9.

## Cấu trúc dự án

```
NhanDangChuSo/
│
├── data/                   # Thư mục chứa dữ liệu (sẽ được tạo tự động)
│
├── models/                 # Thư mục lưu trữ mô hình (sẽ được tạo tự động)
│   ├── best_model.keras    # Mô hình tốt nhất
│   ├── model_config.pkl    # Cấu hình mô hình
│   └── training_history.pkl# Lịch sử quá trình huấn luyện
│
├── results/                # Thư mục chứa kết quả (sẽ được tạo tự động)
│   ├── accuracy_by_digit.png       # Biểu đồ độ chính xác theo chữ số
│   ├── class_distribution.png      # Phân phối của các lớp
│   ├── confusion_matrix.png        # Ma trận nhầm lẫn
│   ├── mnist_samples.png           # Mẫu dữ liệu MNIST
│   ├── model_architecture.png      # Kiến trúc mô hình
│   ├── prediction_examples.png     # Ví dụ dự đoán
│   ├── prediction_slide.png        # Slide dự đoán
│   ├── training_history.png        # Lịch sử huấn luyện
│   ├── tsne_visualization.png      # Trực quan hóa t-SNE
│   ├── evaluation_results.txt      # Kết quả đánh giá
│   └── summary_report.txt          # Báo cáo tổng hợp
│
├── src/                    # Mã nguồn
│   ├── prepare_data.py     # Chuẩn bị dữ liệu
│   ├── train_model.py      # Huấn luyện mô hình
│   ├── evaluate_model.py   # Đánh giá mô hình
│   └── visualize_results.py# Trực quan hóa kết quả
│
├── run.py                  # Script chạy toàn bộ quy trình
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Hướng dẫn sử dụng
```

## Yêu cầu

- Python 3.8 trở lên
- Các thư viện được liệt kê trong `requirements.txt`

## Cài đặt

1. Clone hoặc tải xuống dự án

2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử dụng

Để chạy toàn bộ quy trình (từ chuẩn bị dữ liệu đến trực quan hóa kết quả):

```bash
python run.py
```

Hoặc bạn có thể chạy từng bước riêng biệt:

```bash
# Chuẩn bị dữ liệu
python src/prepare_data.py

# Huấn luyện mô hình
python src/train_model.py

# Đánh giá mô hình
python src/evaluate_model.py

# Trực quan hóa kết quả
python src/visualize_results.py
```

## Quy trình hoạt động

1. **Chuẩn bị dữ liệu**: Tải bộ dữ liệu MNIST, chia thành tập huấn luyện, xác thực và kiểm tra, chuẩn hóa và lưu trữ
2. **Huấn luyện mô hình**: Xây dựng và huấn luyện mô hình ANN với các lớp ẩn và kỹ thuật chống overfitting
3. **Đánh giá mô hình**: Đánh giá hiệu suất mô hình trên tập kiểm tra, tạo ma trận nhầm lẫn và các báo cáo phân loại
4. **Trực quan hóa kết quả**: Trực quan hóa kiến trúc mô hình, không gian đặc trưng và kết quả dự đoán, tạo báo cáo tổng hợp

## Cấu hình mô hình

Mô hình ANN được sử dụng có cấu trúc:
- Đầu vào: 784 nơ-ron (hình ảnh MNIST 28x28 được làm phẳng)
- Các lớp ẩn: 2 lớp ẩn với 128 nơ-ron mỗi lớp và hàm kích hoạt ReLU
- Lớp dropout: Tỷ lệ 0.2 sau mỗi lớp ẩn để giảm overfitting
- Đầu ra: 10 nơ-ron với hàm kích hoạt softmax (dự đoán 10 chữ số 0-9)

## Kết quả

Sau khi chạy xong quy trình, các kết quả sau sẽ được tạo ra trong thư mục `results/`:
- Biểu đồ quá trình huấn luyện
- Ma trận nhầm lẫn
- Ví dụ dự đoán
- Biểu đồ độ chính xác theo chữ số
- Báo cáo tổng hợp và nhiều trực quan khác

## Lưu ý
Do kích thước folder Data quá lớn nên mình đã không push lên. Tuy nhiên, khi chạy mô hình thì bộ dữ liệu và folder sẽ được tự động tạo nhé!
