"""
Huấn luyện mô hình ANN cho bài toán nhận dạng chữ số viết tay.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time
import pickle

def create_ann_model(input_shape, hidden_layers, neurons_per_layer, dropout_rate=0.2):
    """Tạo mô hình ANN với các tham số cấu hình"""
    model = Sequential()
    
    # Lớp đầu vào
    model.add(Dense(neurons_per_layer, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    
    # Thêm các lớp ẩn
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Lớp đầu ra: 10 lớp cho 10 chữ số
    model.add(Dense(10, activation='softmax'))
    
    # Biên dịch mô hình
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Lấy đường dẫn đến thư mục gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    print("Đang tải dữ liệu...")
    
    # Tải dữ liệu từ thư mục data
    data_dir = os.path.join(project_dir, "data")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    # Chuyển đổi nhãn thành one-hot encoding
    y_train_categorical = to_categorical(y_train, 10)
    y_val_categorical = to_categorical(y_val, 10)
    
    print(f"Kích thước dữ liệu huấn luyện: {X_train.shape}")
    print(f"Kích thước dữ liệu xác thực: {X_val.shape}")
    
    # Thiết lập thông số mô hình
    input_shape = X_train.shape[1]  # số lượng feature (784 cho ảnh MNIST 28x28)
    hidden_layers = 2                # số lượng lớp ẩn
    neurons_per_layer = 128          # số lượng nơ-ron mỗi lớp
    dropout_rate = 0.2               # tỷ lệ dropout để giảm overfitting
    batch_size = 128                 # kích thước batch
    epochs = 30                      # số lượng epoch tối đa
    
    # Tạo thư mục để lưu mô hình
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Tạo callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(models_dir, "best_model.keras"),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Tạo và huấn luyện mô hình
    model = create_ann_model(
        input_shape=input_shape,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        dropout_rate=dropout_rate
    )
    
    # Hiển thị thông tin mô hình
    model.summary()
    
    # Bắt đầu thời gian huấn luyện
    start_time = time.time()
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train_categorical,
        validation_data=(X_val, y_val_categorical),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Tính thời gian huấn luyện
    training_time = time.time() - start_time
    print(f"Thời gian huấn luyện: {training_time:.2f} giây")
    
    # Lưu lịch sử huấn luyện
    with open(os.path.join(models_dir, "training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)
    
    # Vẽ biểu đồ accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Huấn luyện')
    plt.plot(history.history['val_accuracy'], label='Xác thực')
    plt.title('Độ chính xác theo epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Độ chính xác')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Huấn luyện')
    plt.plot(history.history['val_loss'], label='Xác thực')
    plt.title('Mất mát theo epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mất mát')
    plt.legend()
    plt.grid(True)
    
    # Lưu biểu đồ quá trình huấn luyện
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    plt.close()
    
    print("Đã lưu biểu đồ quá trình huấn luyện tại 'results/training_history.png'")
    
    # Lưu thông tin cấu hình mô hình
    model_config = {
        "input_shape": input_shape,
        "hidden_layers": hidden_layers,
        "neurons_per_layer": neurons_per_layer,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "epochs_trained": len(history.history['accuracy']),
        "training_time": training_time,
        "final_accuracy": history.history['accuracy'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1]
    }
    
    with open(os.path.join(models_dir, "model_config.pkl"), "wb") as f:
        pickle.dump(model_config, f)
    
    print("Đã lưu cấu hình mô hình tại 'models/model_config.pkl'")
    print(f"Đã lưu mô hình tốt nhất tại 'models/best_model.keras'")
    
    return True

if __name__ == "__main__":
    main() 