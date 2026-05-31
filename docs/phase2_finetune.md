# Tài liệu Phase 2: Fine-tuning & Data Prep

Tài liệu này đặc tả chi tiết về công cụ, tham số và phương pháp sử dụng trong **Phase 2: Fine-tuning** để huấn luyện tinh chỉnh mô hình YOLOv8 trên tập dữ liệu chuyên biệt.

---

## 1. Công cụ sử dụng (Tools)
- **Framework huấn luyện:** `Ultralytics YOLOv8` (sử dụng phương thức `model.train()`).
- **Môi trường tính toán:** PyTorch tự động phát hiện và sử dụng card đồ họa NVIDIA (CUDA) nếu có driver phù hợp để đẩy nhanh tốc độ train, ngược lại sẽ sử dụng CPU.

---

## 2. Cấu hình Dữ liệu (`configs/traffic.yaml`)
Mô hình nhận dạng thông tin cấu trúc dữ liệu thông qua tệp cấu hình yaml với 4 nhãn lớp được định nghĩa chuẩn xác:
```yaml
path: d:/HK8/DL/data/dataset # Thư mục gốc dữ liệu
train: train/images          # Đường dẫn ảnh train
val: valid/images            # Đường dẫn ảnh validation
test: test/images            # Đường dẫn ảnh test (tùy chọn)

nc: 4                        # Số lượng lớp đối tượng
names:                       # Ánh xạ nhãn lớp
  0: car
  1: motor
  2: truck
  3: bus
```

---

## 3. Các tham số huấn luyện cơ bản (Hyperparameters)

| Tham số | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--config` | `configs/traffic.yaml` | Đường dẫn tới tệp cấu hình dữ liệu. |
| `--model` | `yolov8n.pt` | Trọng số mô hình pre-trained làm điểm bắt đầu huấn luyện. |
| `--epochs` | `30` | Số vòng lặp huấn luyện qua toàn bộ tập dữ liệu (Epochs). |
| `--batch` | `8` | Kích thước lô dữ liệu xử lý trong một bước (Batch size). |
| `--optimizer` | `SGD` | Bộ tối ưu hóa sử dụng (`SGD` hoặc `Adam`). |
| `--cos-lr` | `False` | Sử dụng Cosine Learning Rate Decay hay không. |
| `--workers` | `0` | Số lượng luồng nạp dữ liệu (đặt bằng 0 trên Windows để tránh lỗi đông cứng). |

---

## 4. Các lệnh đã thực hiện (Execution Commands)

### A. Lệnh chạy huấn luyện mô hình Fine-tuned (SGD, 30 epochs)
Chạy huấn luyện mô hình YOLOv8n trên tập dữ liệu tùy chỉnh bằng bộ tối ưu hóa SGD trong 30 epochs:
```bash
.\venv\Scripts\python src/train.py --epochs 30 --batch 8 --optimizer SGD --workers 0
```
*Trọng số tốt nhất được lưu tại:* `d:\HK8\DL\runs\train\yolov8_sgd_cos_false\weights\best.pt`

### B. Lệnh chạy đánh giá mô hình Base so với Fine-tuned
Chạy đánh giá mô hình pre-trained COCO gốc (`yolov8n.pt`) trên tập dữ liệu validation tùy chỉnh và in bảng so sánh đối chiếu trực tiếp với mô hình Fine-tuned SGD:
```bash
.\venv\Scripts\python src/evaluate_base.py
```

---

## 5. Bảng so sánh thông số (Base vs Fine-tuned SGD)

Dưới đây là bảng so sánh hiệu năng trực quan giữa mô hình Base Pre-trained (COCO) và mô hình sau khi Fine-tune (SGD, 30 Epochs) trên tập dữ liệu validation (156 ảnh):

| Class | Base Precision | Base Recall | Fine-tuned (SGD) Prec | Fine-tuned (SGD) Rec | Sự cải thiện (Precision) | Sự cải thiện (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **car** | 0.593 | 0.587 | 0.872 | 0.898 | **+27.9%** | **+31.1%** |
| **motor** | 0.227 | 0.047 | 0.853 | 0.813 | **+62.6%** | **+76.6%** |
| **truck** | 0.253 | 0.108 | 0.734 | 0.879 | **+48.1%** | **+77.1%** |
| **bus** | 0.124 | 0.686 | 0.892 | 0.771 | **+76.8%** | **-0.085%** |
| **all (mAP)** | **0.369** | **0.309** | **0.838** | **0.840** | **+46.9%** | **+53.1%** |

### Phân tích nhanh:
- **Tập dữ liệu tùy chỉnh (Vietnamese Vehicles):** Mang tính đặc trưng đường phố rất cao, đặc biệt lượng xe máy cực lớn và nhiều loại xe tải nhỏ (light/box trucks) mà mô hình COCO thường nhận diện nhầm thành xe bus hoặc bỏ sót hoàn toàn.
- **Sự cải thiện:** Mô hình Fine-tuned tăng đáng kể cả Precision và Recall ở toàn bộ các lớp. Tỷ lệ nhận diện đúng xe máy (`motor`) tăng vọt từ 22.7% Precision / 4.7% Recall lên **85.3% Precision / 81.3% Recall**.
- **Lớp Bus:** Recall của Base Model đạt 68.6% chủ yếu là do nó nhận diện rất nhiều xe tải nhỏ (truck) và một số xe con cỡ lớn thành `bus` (tỉ lệ Precision của Base bus cực kỳ thấp, chỉ 12.4% do dương tính giả quá nhiều). Mô hình Fine-tuned đã sửa lỗi này triệt để, kéo Precision lên **89.2%** và Recall đạt mức thực chất **77.1%**.

---

## 6. Phương pháp huấn luyện (Methodology)

1. **Khởi tạo học chuyển tiếp (Transfer Learning):**
   - Script `src/train.py` load mô hình pre-trained `yolov8n.pt` (huấn luyện trên COCO 80 lớp).
   - Khi nhận file cấu hình `traffic.yaml` với `nc: 4`, YOLOv8 tự động thay thế lớp dự đoán cuối cùng (Classification Head) bằng một lớp mới với 4 đầu ra. Trọng số lớp mới này được khởi tạo ngẫu nhiên.
   - Trọng số trích xuất đặc trưng của Backbone và Neck được giữ lại từ COCO làm nền tảng ban đầu.
2. **Quá trình huấn luyện:**
   - Dữ liệu ảnh được đưa vào mô hình theo từng lô (batch size).
   - Hàm loss tính toán sai lệch giữa dự đoán (bounding box + class) với nhãn thực tế (ground truth).
   - Bộ tối ưu hóa (Optimizer, mặc định là SGD) cập nhật các trọng số của mô hình để giảm thiểu loss.
3. **Lưu Checkpoints:**
   - Sau mỗi epoch, mô hình kiểm tra hiệu năng trên tập validation.
   - Checkpoint tốt nhất (best.pt) và checkpoint cuối cùng (last.pt) sẽ tự động được lưu trữ tại thư mục: `runs/train/yolov8_[optimizer]_cos_[boolean]/weights/`.

