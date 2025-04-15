"""
Đánh giá mô hình ANN cho bài toán nhận dạng chữ số viết tay.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, results_dir):
    """Vẽ ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Ma trận nhầm lẫn')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    print("Đã lưu ma trận nhầm lẫn tại 'results/confusion_matrix.png'")

def plot_example_predictions(X, y_true, y_pred, n_examples=25, results_dir=None):
    """Vẽ một số ví dụ dự đoán (bao gồm cả đúng và sai)."""
    # Chuyển đổi dữ liệu phẳng về hình ảnh 28x28
    images = X.reshape(-1, 28, 28)
    
    # Tính toán các vị trí đúng và sai
    correct_idxs = np.where(y_true == y_pred)[0]
    incorrect_idxs = np.where(y_true != y_pred)[0]
    
    # Chọn ngẫu nhiên các ví dụ
    n_correct = min(n_examples // 2, len(correct_idxs))
    n_incorrect = min(n_examples - n_correct, len(incorrect_idxs))
    
    correct_samples = np.random.choice(correct_idxs, n_correct, replace=False)
    incorrect_samples = np.random.choice(incorrect_idxs, n_incorrect, replace=False)
    
    # Tạo lưới để hiển thị
    n_cols = 5
    n_rows = int(np.ceil((n_correct + n_incorrect) / n_cols))
    
    plt.figure(figsize=(15, 3*n_rows))
    
    # Hiển thị các mẫu dự đoán chính xác (màu xanh)
    for i, idx in enumerate(correct_samples):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Thực: {y_true[idx]}, Dự đoán: {y_pred[idx]}", color='green')
        plt.axis('off')
    
    # Hiển thị các mẫu dự đoán sai (màu đỏ)
    for i, idx in enumerate(incorrect_samples):
        plt.subplot(n_rows, n_cols, n_correct+i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Thực: {y_true[idx]}, Dự đoán: {y_pred[idx]}", color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prediction_examples.png"))
    plt.close()
    
    print("Đã lưu hình ảnh các ví dụ dự đoán tại 'results/prediction_examples.png'")

def main():
    # Lấy đường dẫn đến thư mục gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")
    
    # Đảm bảo thư mục kết quả tồn tại
    os.makedirs(results_dir, exist_ok=True)
    
    print("Đang tải dữ liệu kiểm tra và mô hình...")
    
    # Tải dữ liệu kiểm tra
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Tải mô hình
    model = load_model(os.path.join(models_dir, "best_model.keras"))
    
    # Dự đoán
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Đánh giá độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác trên tập kiểm tra: {accuracy:.4f}")
    
    # Tạo báo cáo phân loại chi tiết
    class_names = [str(i) for i in range(10)]
    classification_rep = classification_report(y_test, y_pred, target_names=class_names)
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_rep)
    
    # Lưu báo cáo đánh giá
    with open(os.path.join(results_dir, "evaluation_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Độ chính xác trên tập kiểm tra: {accuracy:.4f}\n\n")
        f.write("Báo cáo phân loại chi tiết:\n")
        f.write(classification_rep)
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(y_test, y_pred, class_names, results_dir)
    
    # Vẽ một số ví dụ dự đoán
    plot_example_predictions(X_test, y_test, y_pred, results_dir=results_dir)
    
    # Phân tích độ chính xác theo từng chữ số
    accuracies_by_digit = []
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        if len(digit_indices) > 0:
            digit_accuracy = accuracy_score(y_test[digit_indices], y_pred[digit_indices])
            accuracies_by_digit.append(digit_accuracy)
        else:
            accuracies_by_digit.append(0)
    
    # Vẽ biểu đồ độ chính xác theo chữ số
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), accuracies_by_digit, color='skyblue')
    plt.xlabel('Chữ số')
    plt.ylabel('Độ chính xác')
    plt.title('Độ chính xác phân loại theo từng chữ số')
    plt.xticks(range(10))
    plt.ylim(0, 1.1)
    for i, acc in enumerate(accuracies_by_digit):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_by_digit.png"))
    plt.close()
    
    print("Đã lưu biểu đồ độ chính xác theo chữ số tại 'results/accuracy_by_digit.png'")
    
    return True

if __name__ == "__main__":
    main() 