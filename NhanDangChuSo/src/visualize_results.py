"""
Trực quan hóa kết quả và tạo báo cáo tổng hợp cho bài toán nhận dạng chữ số viết tay.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_model_architecture(model, results_dir):
    """Trực quan hóa kiến trúc mô hình."""
    try:
        plot_model(
            model,
            to_file=os.path.join(results_dir, "model_architecture.png"),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True
        )
        print("Đã lưu sơ đồ kiến trúc mô hình tại 'results/model_architecture.png'")
    except Exception as e:
        print(f"Lỗi khi vẽ kiến trúc mô hình: {e}")
        print("Lưu ý: Bạn cần cài đặt graphviz để vẽ kiến trúc mô hình.")

def visualize_feature_space(X, y, perplexity=30, n_samples=2000, results_dir=None):
    """Trực quan hóa không gian đặc trưng bằng t-SNE."""
    print("Đang thực hiện giảm chiều dữ liệu với t-SNE (có thể mất vài phút)...")
    
    # Chọn ngẫu nhiên một số mẫu để giảm thời gian tính toán
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Áp dụng giảm chiều t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Vẽ biểu đồ phân cụm
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        X_tsne[:, 0], 
        X_tsne[:, 1], 
        c=y_sample, 
        cmap='tab10',
        alpha=0.7,
        s=50
    )
    plt.colorbar(scatter, ticks=range(10), label='Chữ số')
    plt.title('Trực quan hóa không gian đặc trưng với t-SNE')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(os.path.join(results_dir, "tsne_visualization.png"))
    plt.close()
    
    print("Đã lưu trực quan hóa t-SNE tại 'results/tsne_visualization.png'")

def create_summary_report(project_dir, results_dir, models_dir):
    """Tạo báo cáo tổng hợp dựa trên kết quả đã có."""
    # Đọc thông tin cấu hình mô hình
    try:
        with open(os.path.join(models_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)
        
        # Đọc lịch sử huấn luyện
        with open(os.path.join(models_dir, "training_history.pkl"), "rb") as f:
            history = pickle.load(f)
        
        # Đọc kết quả đánh giá
        with open(os.path.join(results_dir, "evaluation_results.txt"), "r", encoding="utf-8") as f:
            evaluation_results = f.read()
        
        # Tạo báo cáo tổng hợp
        with open(os.path.join(results_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
            f.write("BÁO CÁO TỔNG HỢP NHẬN DẠNG CHỮ SỐ VIẾT TAY VỚI ANN\n")
            f.write("=" * 80 + "\n\n")
            
            # Thông tin cấu hình mô hình
            f.write("CẤU HÌNH MÔ HÌNH\n")
            f.write("-" * 80 + "\n")
            f.write(f"Số lớp ẩn: {model_config['hidden_layers']}\n")
            f.write(f"Số nơ-ron mỗi lớp: {model_config['neurons_per_layer']}\n")
            f.write(f"Dropout rate: {model_config['dropout_rate']}\n")
            f.write(f"Batch size: {model_config['batch_size']}\n")
            f.write(f"Số epoch đã huấn luyện: {model_config['epochs_trained']}\n")
            f.write(f"Thời gian huấn luyện: {model_config['training_time']:.2f} giây\n\n")
            
            # Kết quả huấn luyện
            f.write("KẾT QUẢ HUẤN LUYỆN\n")
            f.write("-" * 80 + "\n")
            f.write(f"Độ chính xác trên tập huấn luyện: {model_config['final_accuracy']:.4f}\n")
            f.write(f"Độ chính xác trên tập xác thực: {model_config['final_val_accuracy']:.4f}\n\n")
            
            # Kết quả đánh giá
            f.write("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM TRA\n")
            f.write("-" * 80 + "\n")
            f.write(evaluation_results + "\n\n")
            
            # Hình ảnh kết quả
            f.write("DANH SÁCH CÁC HÌNH ẢNH KẾT QUẢ\n")
            f.write("-" * 80 + "\n")
            for file in os.listdir(results_dir):
                if file.endswith((".png", ".jpg")):
                    f.write(f"- {file}\n")
        
        print("Đã tạo báo cáo tổng hợp tại 'results/summary_report.txt'")
    
    except Exception as e:
        print(f"Lỗi khi tạo báo cáo tổng hợp: {e}")

def main():
    # Lấy đường dẫn đến thư mục gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")
    
    # Đảm bảo thư mục kết quả tồn tại
    os.makedirs(results_dir, exist_ok=True)
    
    print("Đang tải dữ liệu và mô hình để trực quan hóa...")
    
    # Tải dữ liệu kiểm tra
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Tải mô hình
    model = load_model(os.path.join(models_dir, "best_model.keras"))
    
    # Trực quan hóa kiến trúc mô hình
    visualize_model_architecture(model, results_dir)
    
    # Trực quan hóa không gian đặc trưng
    visualize_feature_space(X_test, y_test, results_dir=results_dir)
    
    # Tạo báo cáo tổng hợp
    create_summary_report(project_dir, results_dir, models_dir)
    
    # Tạo slide nhận dạng trực quan với mô hình
    plt.figure(figsize=(15, 8))
    
    # Chọn một số mẫu ngẫu nhiên để dự đoán
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    X_samples = X_test[sample_indices]
    y_true = y_test[sample_indices]
    
    # Dự đoán
    y_pred_prob = model.predict(X_samples)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Vẽ hình ảnh và kết quả dự đoán
    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        # Chuyển vector thành ảnh
        img = X_samples[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        # Lấy xác suất dự đoán cho chữ số thực tế
        true_prob = y_pred_prob[i, y_true[i]]
        
        # Tiêu đề màu xanh nếu đúng, đỏ nếu sai
        color = 'green' if y_pred[i] == y_true[i] else 'red'
        plt.title(f"Thực tế: {y_true[i]}\nDự đoán: {y_pred[i]}\nTự tin: {true_prob:.2f}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prediction_slide.png"))
    plt.close()
    
    print("Đã lưu slide trực quan kết quả nhận dạng tại 'results/prediction_slide.png'")
    
    return True

if __name__ == "__main__":
    main() 