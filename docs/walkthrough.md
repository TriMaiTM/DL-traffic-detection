# Hướng dẫn chạy và Nhật ký kết quả dự án (Walkthrough)

Tài liệu này tổng hợp toàn bộ quy trình thiết lập, kịch bản lệnh chạy qua các Phase và kết quả đạt được của dự án nhận diện, phân loại và theo dõi phương tiện giao thông tại Việt Nam sử dụng YOLOv8.

---

## 1. Cấu trúc thư mục dự án

```text
d:/HK8/DL/
├── configs/
│   └── traffic.yaml             # Cấu hình dataset (4 lớp: car, motor, truck, bus)
├── data/
│   ├── dataset/                 # Thư mục chứa ảnh train/val/test sau khi giải nén
│   ├── input/                   # Chứa video giao thông đầu vào (test.mp4, hanoi_traffic.mp4...)
│   └── output/                  # Chứa video kết quả đầu ra (tracking, baseline...)
├── docs/
│   ├── datasets.md              # Chi tiết tập dữ liệu
│   ├── phase1_base.md           # Nhật ký Phase 1 (Baseline)
│   ├── phase2_finetune.md       # Nhật ký Phase 2 (Fine-tuning)
│   └── optimization.md          # Phân tích so sánh tối ưu hóa ở Phase 3
├── runs/
│   └── train/                   # Thư mục lưu kết quả huấn luyện (weights, plots...)
├── src/
│   ├── baseline.py              # Script chạy baseline COCO (Phase 1)
│   ├── train.py                 # Script huấn luyện fine-tune (Phase 2 & 3)
│   ├── evaluate_base.py         # Script đánh giá mô hình Base so với Fine-tuned SGD
│   ├── evaluate_all_models.py   # Script đánh giá so sánh tương quan 3 mô hình (Phase 3)
│   ├── plot_results.py          # Script vẽ biểu đồ so sánh Loss & mAP
│   └── tracker.py               # Script theo dõi tracking cao cấp (Phase 4)
└── walkthrough.md               # Tài liệu tổng kết dự án (File này)
```

---

## 2. Hướng dẫn cài đặt và thiết lập môi trường

Môi trường đã được thiết lập sẵn trong Virtual Environment (`venv`) tại thư mục dự án với Python 3.13.1 và PyTorch hỗ trợ GPU CUDA:
```powershell
# Kích hoạt môi trường ảo (nếu chưa kích hoạt)
.\venv\Scripts\activate
```

---

## 3. Hướng dẫn chạy chi tiết qua các Phase

### Phase 1: Baseline Setup & Inference (Mô hình gốc COCO)
Chạy thử nghiệm phát hiện phương tiện gốc (chỉ lọc 3 nhãn xe từ COCO):
```powershell
.\venv\Scripts\python src/baseline.py --video data/input/test.mp4 --output data/output/baseline.mp4
```

### Phase 2: Fine-tuning (Huấn luyện tinh chỉnh)
Huấn luyện mô hình YOLOv8n trên tập dữ liệu giao thông Việt Nam (4 lớp nhãn) bằng bộ tối ưu hóa SGD trong 30 epochs:
```powershell
.\venv\Scripts\python src/train.py --epochs 30 --batch 8 --optimizer SGD --workers 0
```
*Kết quả lưu tại:* `runs/train/yolov8_sgd_cos_false/`

### Phase 3: Thử nghiệm so sánh & Tối ưu hóa (Optimization)
Để chọn ra cấu hình huấn luyện tối ưu nhất, chúng ta thực hiện thêm 2 lượt train thử nghiệm:

1. **Lượt 2: Chạy với bộ tối ưu hóa `Adam`:**
   ```powershell
   .\venv\Scripts\python src/train.py --epochs 30 --batch 8 --optimizer Adam --workers 0
   ```
2. **Lượt 3: Chạy với bộ tối ưu hóa `SGD` và `Cosine Learning Rate Decay`:**
   ```powershell
   .\venv\Scripts\python src/train.py --epochs 30 --batch 8 --optimizer SGD --cos-lr --workers 0
   ```

3. **Vẽ biểu đồ so sánh tự động:**
   ```powershell
   .\venv\Scripts\python src/plot_results.py
   ```
   *Biểu đồ lưu tại:* [comparison_plots.png](file:///d:/HK8/DL/runs/train/comparison_plots.png)

4. **Trích xuất so sánh chi tiết tất cả các chỉ số (Precision, Recall, F1, mAP):**
   ```powershell
   .\venv\Scripts\python src/evaluate_all_models.py
   ```
   *Tài liệu phân tích tương quan chi tiết xem tại:* [docs/optimization.md](file:///d:/HK8/DL/docs/optimization.md)

### Phase 4: Output & Tracking Integration (Theo dõi đối tượng)
Sử dụng trọng số tốt nhất (`yolov8_sgd_cos_true`) để chạy tracking kèm bảng đếm xe thời gian thực (Live Counter) và lọc nhiễu tòa nhà bằng vùng ROI:

```powershell
# Chạy tracking mặc định trên video TPHCM (Không vẽ đường ranh giới ROI trên màn hình cho video đẹp hơn)
# Mặc định: imgsz=960, conf=0.2, iou=0.7, roi-y=0.35 (bỏ qua 35% phía trên để loại bỏ nhiễu tòa nhà/bầu trời)
.\venv\Scripts\python src/tracker.py --video data/input/tphcm_traffic.mp4 --output data/output/tracking_tphcm.mp4

# Chạy hiển thị rõ đường ROI màu đỏ (để kiểm tra vùng quét)
.\venv\Scripts\python src/tracker.py --video data/input/tphcm_traffic.mp4 --output data/output/tracking_tphcm_show_roi.mp4 --show-roi
```

---

## 4. Kết quả tổng hợp & Đánh giá dự án

### Bảng so sánh hiệu năng các mô hình thử nghiệm:
| Mô hình thử nghiệm | Precision | Recall (Độ phủ) | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COCO Base Pre-trained** | 0.369 | 0.309 | 0.336 | - | - |
| **SGD + No Cosine** | 0.838 | 0.840 | 0.839 | 0.899 | 0.601 |
| **Adam + No Cosine** | 0.742 | 0.805 | 0.772 | 0.843 | 0.558 |
| **SGD + Cosine Decay** | 0.826 | **0.862** | **0.843** | **0.921** | **0.617** |

### Kết luận quan trọng:
*   Mô hình **SGD + Cosine Decay** đạt chất lượng vượt trội nhất với **mAP@0.5 = 92.1%** và **Recall = 86.2%**. Việc hạ tốc độ học mịn hơn ở các epoch cuối cùng giúp mô hình nhận diện chính xác các chi tiết nhỏ của đám đông xe máy và các lớp dữ liệu hiếm như xe buýt.
*   **Giải pháp ROI lọc hậu cảnh:** Thêm chức năng lọc tọa độ Y (`--roi-y 0.35`) giúp triệt tiêu hoàn toàn lỗi nhận diện nhầm các khối cửa sổ của tòa nhà cao tầng thành thùng xe tải (Truck).
*   **Giải pháp nâng cao IoU NMS:** Tăng ngưỡng `--iou` lên `0.7` giúp giữ lại các xe máy đi sát nhau trong đám đông mà không bị bộ lọc trùng lặp NMS xóa bỏ nhầm.
