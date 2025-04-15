import os
import subprocess
import time

def run_script(script_path, description):
    """Chạy một script Python và hiển thị thông báo"""
    print("\n" + "="*80)
    print(f"ĐANG THỰC HIỆN: {description}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Lấy đường dẫn hiện tại của file run.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Tạo đường dẫn đầy đủ đến script cần chạy
    full_script_path = os.path.join(current_dir, script_path)
    
    # Chạy script
    result = subprocess.run(['python', full_script_path], check=False)
    
    if result.returncode != 0:
        print(f"\nLỖI: Script '{script_path}' không thực hiện thành công.")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\nHoàn thành trong {elapsed_time:.2f} giây.\n")
    return True

def main():
    """Chạy toàn bộ quy trình"""
    # Danh sách các bước sẽ thực hiện
    steps = [
        ("src/prepare_data.py", "Chuẩn bị dữ liệu MNIST"),
        ("src/train_model.py", "Huấn luyện mô hình ANN"),
        ("src/evaluate_model.py", "Đánh giá mô hình chi tiết"),
        ("src/visualize_results.py", "Trực quan hóa kết quả")
    ]
    
    # Lấy đường dẫn hiện tại của file run.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tạo các thư mục cần thiết
    os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    
    # Chuyển đến thư mục của file run.py
    os.chdir(current_dir)
    
    # Thực hiện từng bước
    for script_path, description in steps:
        success = run_script(script_path, description)
        if not success:
            print(f"Quy trình bị dừng tại bước: {description}")
            break
    
    print("\n" + "="*80)
    print("KẾT QUẢ THỰC HIỆN")
    print("="*80)
    
    # Hiển thị kết quả cuối cùng
    results_dir = os.path.join(current_dir, "results")
    evaluation_file = os.path.join(results_dir, "evaluation_results.txt")
    
    if os.path.exists(evaluation_file):
        print("\nKết quả đánh giá:")
        with open(evaluation_file, 'r', encoding="utf-8") as f:
            print(f.read())
    
    print("\nCác kết quả trực quan được lưu trong thư mục 'results/':")
    for file in os.listdir(results_dir):
        if file.endswith((".png", ".jpg")):
            print(f"  - {file}")
    
    print("\nQuy trình hoàn tất!")

if __name__ == "__main__":
    main() 