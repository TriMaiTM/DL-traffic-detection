# Hệ thống phát hiện & phân loại phương tiện giao thông trên video

Dự án xây dựng hệ thống phát hiện, phân loại (Ô tô, Xe máy, Xe tải) và theo dõi (Tracking) phương tiện giao thông trên video, thực hiện so sánh đánh giá hiệu năng trước/sau khi fine-tuning và tối ưu hóa hyperparameter.

## User Review Required

> [!IMPORTANT]
> **1. Lựa chọn Model & Framework:** Đề xuất sử dụng **YOLOv8** (hoặc YOLOv11) từ Ultralytics. Lý do:
> - Có sẵn pre-trained weights mạnh mẽ trên COCO (chứa sẵn car, motorcycle, truck).
> - Tích hợp sẵn bộ theo dõi tracking (ByteTrack, BoT-SORT) chỉ bằng 1 dòng code.
> - Hỗ trợ API Python cực kỳ tường minh cho việc fine-tune, tùy chỉnh optimizer (SGD/Adam), learning rate scheduler (cosine), và data augmentation.
>
> **2. Lựa chọn Dataset cho Fine-tuning:**
> - Chúng ta sẽ sử dụng một bộ dataset chuyên biệt chứa chính xác 3 class: `car`, `motorcycle`, `truck`.
> - Dữ liệu chi tiết về số lượng ảnh, phân chia Train/Val/Test, và định dạng nhãn đã được đặc tả chi tiết trong file tài liệu mới lập [datasets.md](file:///d:/HK8/DL/docs/datasets.md).

## Proposed Changes

Chúng ta sẽ tổ chức mã nguồn theo cấu trúc thư mục rõ ràng như sau:
```text
d:/HK8/DL/
├── docs/                   # Tài liệu lý thuyết và đặc tả dataset
│   └── datasets.md         # Chi tiết về tập dữ liệu COCO & Custom Dataset
├── data/                   # Chứa dữ liệu train/val/test, input và output video
│   ├── input/              # Video đầu vào kiểm thử (ví dụ: test.mp4)
│   ├── output/             # Video đầu ra sau xử lý (ví dụ: baseline.mp4)
│   └── dataset/            # Ảnh và nhãn (yolo format) cho fine-tuning
├── src/                    # Mã nguồn chính
│   ├── baseline.py         # Chạy pre-trained YOLO trên video mẫu (Phase 1)
│   ├── train.py            # Script train/fine-tune hỗ trợ cấu hình optimizer, LR scheduler (Phase 2 & 3)
│   ├── tracker.py          # Chạy inference kèm tracking và xuất video (Phase 4)
│   └── utils.py            # Hàm vẽ biểu đồ, tính toán metrics phụ trợ
└── configs/                # File cấu hình yaml cho training và dataset
```

---

### Phase 1: Base (Baseline Setup & Inference)
#### [NEW] [baseline.py](file:///d:/HK8/DL/src/baseline.py)
- Load model pre-trained YOLOv8 (ví dụ: `yolov8n.pt` hoặc `yolov8s.pt` đã train trên COCO 80 classes).
- Thực hiện detect và lọc ra đúng 3 class chỉ mục của COCO: `Index 2` (car), `Index 3` (motorcycle), `Index 7` (truck) trên video kiểm thử tại `data/input/`.
- Vẽ bounding box, label và xuất ra video baseline vào `data/output/` để đánh giá chất lượng nhận diện ban đầu.

---

### Phase 2: Fine-tuning & Data Prep
#### [NEW] [train.py](file:///d:/HK8/DL/src/train.py)
- Nhận cấu hình từ file `configs/traffic.yaml` trỏ đến tập dữ liệu custom có 3 lớp.
- **Cơ chế Transfer Learning:** Khi bắt đầu train từ checkpoint pre-trained (`yolov8n.pt`), mô hình sẽ tự động cắt bỏ lớp classification cũ (80 lớp) và thay thế bằng lớp phân loại mới (3 lớp). Trọng số phần trích xuất đặc trưng (backbone) được giữ lại để tiếp tục tối ưu hóa.
- Tùy chỉnh các tham số cơ bản: Learning rate (LR), Batch size, Epochs.

---

### Phase 3: Optimization & Evaluation
#### [MODIFY] [train.py](file:///d:/HK8/DL/src/train.py)
- Bổ sung tham số lựa chọn optimizer: `optimizer='SGD'` hoặc `optimizer='Adam'` (hoặc `AdamW`).
- Bổ sung cấu hình scheduler: `cos_lr=True` để áp dụng Cosine Decay.
- Tùy chỉnh các tham số Data Augmentation trong YOLO (như `mosaic`, `mixup`, `degrees`, `hsv_h/s/v`).
- Vẽ biểu đồ so sánh Loss curve và Precision-Recall curve giữa các lần thử nghiệm (SGD vs Adam).

---

### Phase 4: Output & Tracking Integration
#### [NEW] [tracker.py](file:///d:/HK8/DL/src/tracker.py)
- Load model sau khi đã fine-tune và tối ưu tốt nhất.
- Chạy inference trên video test tích hợp bộ tracking (ByteTrack hoặc BoT-SORT).
- Xuất video output cuối cùng chứa: Bounding box sắc nét + Label phân loại đúng + ID của từng phương tiện được theo dõi qua các frame.

## Verification Plan

### Automated Tests & Runs
- **Verify Phase 1:** Chạy lệnh `python src/baseline.py --video data/input/test.mp4` để kiểm tra khả năng đọc video, detect bằng model pre-trained và ghi video đầu ra thành công.
- **Verify Phase 2 & 3:** Chạy thử nghiệm training ngắn hạn (ví dụ: 2 epochs) bằng lệnh:
  `python src/train.py --epochs 2 --batch 8 --optimizer SGD` và `python src/train.py --epochs 2 --batch 8 --optimizer Adam` để đảm bảo hệ thống lưu log và so sánh được optimizer mà không gặp lỗi phần cứng/phần mềm.
- **Verify Phase 4:** Chạy tracking trên video test và kiểm tra xem video output có xuất hiện ID tracking ổn định không.

### Manual Verification
- Xem trực tiếp video kết quả để đánh giá độ giật lag của bounding box, độ ổn định của ID tracking khi xe bị che khuất.
- Quan sát biểu đồ loss và PR curve để đánh giá xem mô hình có bị overfit hay không.
