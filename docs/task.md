# Danh sách công việc (Task List)

Dưới đây là danh sách các bước cần thực hiện để hoàn thành dự án. Chúng ta sẽ đánh dấu hoàn thành từng mục khi triển khai.

- [x] **Phase 1: Base (Baseline Setup & Inference)**
  - [x] Khởi tạo cấu trúc thư mục dự án (`data/input`, `data/output`, `src`, `configs`, `docs`)
  - [x] Thiết lập môi trường ảo và cài đặt thư viện cần thiết (PyTorch, Ultralytics YOLOv8, OpenCV, Matplotlib...)
  - [x] Chuẩn bị video kiểm thử mẫu đặt vào `data/input/`
  - [x] Tạo file tài liệu `docs/phase1_base.md` lưu trữ chi tiết phương pháp lọc nhãn, tham số baseline
  - [x] Viết mã nguồn `src/baseline.py` chạy inference model pre-trained YOLOv8 lọc 3 nhãn COCO
  - [x] Thực thi chạy thử nghiệm và kiểm chứng video đầu ra thành công
- [x] **Phase 2: Fine-tuning & Data Prep**
  - [x] Chuẩn bị và giải nén Custom Dataset chuyên biệt (4 lớp: car, motor, truck, bus)
  - [x] Viết file cấu hình dataset `configs/traffic.yaml`
  - [x] Cập nhật tài liệu dataset chi tiết vào `docs/datasets.md`
  - [x] Viết mã nguồn huấn luyện `src/train.py` cho lần fine-tune đầu tiên
  - [x] Chạy huấn luyện fine-tune baseline và kiểm tra checkpoint đầu ra
- [x] **Phase 3: Optimization & Evaluation**
  - [x] Cập nhật `src/train.py` để hỗ trợ chuyển đổi Optimizer (SGD vs Adam), Cosine Decay, và tùy chỉnh Augmentation
  - [x] Viết script `src/plot_results.py` vẽ biểu đồ so sánh Loss curve và mAP@0.5 giữa các lần huấn luyện
  - [x] Thực hiện chạy các lượt huấn luyện so sánh tiếp theo:
    - [x] Lượt 2: Bộ tối ưu Adam (`--optimizer Adam`)
    - [x] Lượt 3: Bộ tối ưu SGD với Cosine LR Decay (`--optimizer SGD --cos-lr`)
  - [x] Tạo file tài liệu `docs/optimization.md` ghi nhận tham số tối ưu và các biểu đồ so sánh
- [x] **Phase 4: Output & Tracking Integration**
  - [x] Viết mã nguồn `src/tracker.py` tích hợp bộ theo dõi tracking (ByteTrack hoặc BoT-SORT)
  - [x] Chạy mô hình tối ưu nhất kèm tracking trên cùng video mẫu ở Phase 1
  - [x] Xuất video kết quả hoàn chỉnh với Bounding Box + Label + Tracking ID
  - [x] Viết file tổng kết `walkthrough.md` hướng dẫn chi tiết cách chạy toàn bộ dự án


