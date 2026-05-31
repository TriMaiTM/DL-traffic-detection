# Phát Hiện, Phân Loại và Theo Dõi Phương Tiện Giao Thông Việt Nam (YOLOv8 + ByteTrack)

Dự án môn học Deep Learning: Xây dựng hệ thống phát hiện, phân loại và theo dõi (tracking) 4 lớp phương tiện giao thông chính (**Car, Motor, Truck, Bus**) trên đường phố Việt Nam từ camera giám sát thời gian thực.

---

## 🌟 Tính Năng Nổi Bật
- **Học chuyển tiếp (Transfer Learning):** Tinh chỉnh mô hình YOLOv8-Nano trên tập dữ liệu đặc thù giao thông Việt Nam.
- **Tối ưu hóa đa thực nghiệm:** So sánh hiệu năng giữa các bộ tối ưu (`SGD`, `Adam`) và cơ chế suy giảm tốc độ học (`Cosine Learning Rate Decay`).
- **Theo dõi đối tượng thời gian thực:** Tích hợp thuật toán **ByteTrack** gán ID duy nhất cho từng phương tiện.
- **Bảng thống kê đếm xe (Live Traffic Counter):** Dashboard hiển thị số lượng chi tiết từng loại xe đang xuất hiện trên màn hình với hiệu ứng mờ bán trong suốt (glassmorphism style).
- **Vùng quan tâm động (Dynamic ROI):** Bộ lọc giới hạn chiều cao quét (`--roi-y`) giúp triệt tiêu hoàn toàn lỗi nhận diện nhầm các tòa nhà cao tầng/bầu trời thành xe tải.
- **Tối ưu hóa độ phân giải:** Cho phép tùy chỉnh kích thước ảnh đầu vào (`--imgsz`) và ngưỡng NMS (`--iou`) để nhận diện chính xác đám đông xe máy dừng đèn đỏ.

---

## 📂 Cấu Trúc Thư Mục Dự Án
```text
├── configs/
│   └── traffic.yaml             # Cấu hình dataset (4 lớp: car, motor, truck, bus)
├── data/
│   ├── dataset/                 # [Bỏ qua trên Git] Thư mục chứa tập ảnh train/val/test
│   ├── input/                   # [Bỏ qua trên Git] Thư mục chứa video đầu vào
│   └── output/                  # [Bỏ qua trên Git] Thư mục chứa video kết quả tracking
├── docs/
│   ├── datasets.md              # Chi tiết tập dữ liệu
│   ├── phase1_base.md           # Nhật ký Phase 1 (Baseline)
│   ├── phase2_finetune.md       # Nhật ký Phase 2 (Fine-tuning)
│   ├── optimization.md          # Phân tích so sánh tối ưu hóa ở Phase 3
│   └── report.md                # Báo cáo kết quả dự án hoàn chỉnh
├── runs/
│   └── train/                   # Thư mục lưu kết quả huấn luyện (weights, plots...)
│       └── comparison_plots.png # Biểu đồ so sánh Loss & mAP của 3 mô hình
├── src/
│   ├── baseline.py              # Script chạy baseline COCO (Phase 1)
│   ├── train.py                 # Script huấn luyện fine-tune (Phase 2 & 3)
│   ├── evaluate_base.py         # Script đánh giá mô hình Base so với Fine-tuned SGD
│   ├── evaluate_all_models.py   # Script đánh giá so sánh tương quan 3 mô hình (Phase 3)
│   ├── plot_results.py          # Script vẽ biểu đồ so sánh Loss & mAP
│   └── tracker.py               # Script theo dõi tracking cao cấp (Phase 4)
├── .gitignore                   # Cấu hình bỏ qua các file nặng khi push Git
├── requirements.txt             # Danh sách thư viện cần thiết
└── README.md                    # Hướng dẫn dự án (File này)
```

---

## 🛠️ Hướng Dẫn Cài Đặt và Thiết Lập

### 1. Clone dự án và truy cập thư mục
```bash
git clone <link_github_cua_ban>
cd DL
```

### 2. Thiết lập môi trường ảo (Virtual Environment)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux/macOS)
source venv/bin/activate
```

### 3. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy Dự Án

### Bước 1: Chạy thử nghiệm mô hình Baseline (COCO)
Chạy thử nghiệm phát hiện phương tiện gốc (chỉ lọc 3 nhãn xe từ COCO):
```bash
python src/baseline.py --video data/input/test.mp4 --output data/output/baseline.mp4
```

### Bước 2: Huấn luyện tinh chỉnh (Fine-tuning)
Huấn luyện mô hình YOLOv8n trên tập dữ liệu giao thông Việt Nam (4 lớp nhãn) bằng bộ tối ưu hóa SGD trong 30 epochs:
```bash
python src/train.py --epochs 30 --batch 8 --optimizer SGD --workers 0
```

### Bước 3: Huấn luyện thử nghiệm tối ưu hóa
Thực hiện các lượt train thử nghiệm để so sánh:
```bash
# Lượt 2: Chạy với bộ tối ưu hóa Adam
python src/train.py --epochs 30 --batch 8 --optimizer Adam --workers 0

# Lượt 3: Chạy với bộ tối ưu hóa SGD + Cosine Decay
python src/train.py --epochs 30 --batch 8 --optimizer SGD --cos-lr --workers 0

# Vẽ biểu đồ so sánh tự động
python src/plot_results.py

# Đánh giá chi tiết các độ đo (Precision, Recall, F1, mAP)
python src/evaluate_all_models.py
```

### Bước 4: Chạy Tracking & Thống kê phương tiện giao thông (Mô hình tốt nhất)
Chạy thuật toán ByteTrack trên video giao thông sử dụng trọng số tốt nhất (`yolov8_sgd_cos_true`):

```bash
# Chạy tracking mặc định sạch sẽ (Không hiển thị đường biên ROI trên video)
# Mặc định: imgsz=960, conf=0.2, iou=0.7, roi-y=0.35 (loại bỏ nhiễu tòa nhà/bầu trời)
python src/tracker.py --video data/input/tphcm_traffic.mp4 --output data/output/tracking_tphcm.mp4

# Chạy tracking có vẽ đường ranh giới quét ROI màu đỏ để minh họa
python src/tracker.py --video data/input/tphcm_traffic.mp4 --output data/output/tracking_tphcm_show_roi.mp4 --show-roi
```

---

## 📈 Tóm Tắt Kết Quả Thực Nghiệm

Dưới đây là bảng so sánh chỉ số đánh giá tổng thể thu được trên tập Validation (156 ảnh):

| Mô hình thử nghiệm | Precision | Recall (Độ phủ) | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COCO Base Pre-trained** | 0.369 | 0.309 | 0.336 | - | - |
| **SGD + No Cosine** | 0.838 | 0.840 | 0.839 | 0.899 | 0.601 |
| **Adam + No Cosine** | 0.742 | 0.805 | 0.772 | 0.843 | 0.558 |
| **SGD + Cosine Decay** | 0.826 | **0.862** | **0.843** | **0.921** | **0.617** |

### Biểu đồ so sánh quá trình huấn luyện:
![Biểu đồ so sánh hiệu năng](/runs/train/comparison_plots.png)

*Mô hình **SGD + Cosine Decay** đạt hiệu năng cao nhất, giải quyết tốt bài toán nhận diện xe máy mật độ dày đặc và xe buýt, được chọn làm mô hình hoạt động chính thức.*
