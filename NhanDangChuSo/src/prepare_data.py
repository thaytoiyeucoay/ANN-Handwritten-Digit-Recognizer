"""
Chuẩn bị dữ liệu MNIST cho nhận dạng chữ số viết tay.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Lấy đường dẫn đến thư mục gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    print("Đang tải dữ liệu MNIST...")
    
    # Tải bộ dữ liệu MNIST
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    
    # Tách dữ liệu huấn luyện và xác thực
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # Chuẩn hóa dữ liệu (0-255 -> 0-1)
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    # Chuyển đổi hình ảnh từ 2D thành vector 1D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Kích thước dữ liệu:")
    print(f"  - Huấn luyện: {X_train_flat.shape}")
    print(f"  - Xác thực: {X_val_flat.shape}")
    print(f"  - Kiểm tra: {X_test_flat.shape}")
    
    # Hiển thị một số mẫu
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(f"Nhãn: {y_train[i]}")
        plt.axis('off')
    
    # Tạo thư mục để lưu hình ảnh mẫu
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mnist_samples.png"))
    plt.close()
    
    print("Đã lưu hình ảnh mẫu tại 'results/mnist_samples.png'")
    
    # Lưu dữ liệu đã xử lý
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "X_train.npy"), X_train_flat)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val_flat)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test_flat)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    print("Dữ liệu đã được xử lý và lưu vào thư mục 'data/'")
    
    # Thống kê phân phối nhãn
    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=10, rwidth=0.8)
    plt.xlabel('Chữ số')
    plt.ylabel('Số lượng mẫu')
    plt.title('Phân phối các chữ số trong tập huấn luyện')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(results_dir, "class_distribution.png"))
    plt.close()
    
    print("Đã lưu biểu đồ phân phối dữ liệu tại 'results/class_distribution.png'")
    
    return True

if __name__ == "__main__":
    main()