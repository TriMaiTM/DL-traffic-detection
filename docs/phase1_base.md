# Tài liệu Phase 1: Base (Baseline Setup & Inference)

Tài liệu này đặc tả chi tiết về công cụ, tham số và phương pháp sử dụng trong **Phase 1: Base** để chạy thử nghiệm nhận diện phương tiện giao thông trên video mẫu.

---

## 1. Công cụ sử dụng (Tools)
- **Framework phát hiện vật thể:** `Ultralytics YOLOv8` (Phiên bản kiến trúc mới hỗ trợ cả phát hiện và theo dõi vật thể thời gian thực).
- **Mô hình Base (Pre-trained):** `yolov8n.pt` (YOLOv8 Nano - 3.2 triệu tham số). Đây là phiên bản mô hình nhỏ nhẹ nhất, thích hợp chạy thử nghiệm baseline nhanh chóng trên cả CPU và GPU.
- **Thư viện xử lý Video & Ảnh:** `OpenCV (opencv-python)` để đọc/ghi video và thao tác vẽ bounding box lên khung hình.

---

## 2. Các tham số cấu hình (Hyperparameters & Parameters)

| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `model` | `yolov8n.pt` | Tên tệp trọng số pre-trained trên tập dữ liệu MS COCO. |
| `conf` | `0.25` | Ngưỡng tin cậy tối thiểu (Confidence threshold) để mô hình giữ lại phát hiện. |
| `iou` | `0.70` | Ngưỡng IoU cho thuật toán NMS (Non-Maximum Suppression) để loại bỏ các bounding box bị trùng lặp trên cùng một vật thể. |
| `classes` | `[2, 3, 7]` | Danh sách lớp đối tượng cần giữ lại (chỉ lọc Car, Motorcycle, Truck từ 80 lớp COCO). |
| `device` | Tự động | Ưu tiên chạy trên GPU (`cuda` hoặc `mps`) nếu phần cứng hỗ trợ, ngược lại sẽ chạy trên `cpu`. |

---

## 3. Các lớp đối tượng (Labels)
Chúng ta lọc kết quả nhận diện từ mô hình gốc COCO chỉ lấy 3 lớp:
- **Class Index 2:** `car` (ô tô con, xe du lịch)
- **Class Index 3:** `motorcycle` (xe máy, xe mô tô 2 bánh)
- **Class Index 7:** `truck` (xe tải chở hàng, xe tải lớn)

---

## 4. Phương pháp thực hiện (Methodology)

Quy trình xử lý video của script `src/baseline.py` gồm các bước sau:

1. **Khởi tạo:** Load mô hình YOLOv8 với trọng số pre-trained `yolov8n.pt`. Tệp này sẽ tự động được tải từ máy chủ Ultralytics nếu chưa có sẵn ở thư mục dự án.
2. **Đọc Video:** Dùng `cv2.VideoCapture` để mở video đầu vào. Lấy các thông số của video gốc: Chiều rộng (width), Chiều cao (height), Tốc độ khung hình (FPS) để cấu hình cho video đầu ra.
3. **Xử lý từng Frame (Frame-by-frame processing):**
   - Đọc từng khung hình của video.
   - Chạy inference thông qua YOLOv8 với tham số lọc lớp:
     `results = model(frame, conf=0.25, iou=0.7, classes=[2, 3, 7])`
   - Đối với mỗi đối tượng phát hiện được, trích xuất:
     - Tọa độ bounding box `(x1, y1, x2, y2)`.
     - Xác suất tin cậy (confidence score).
     - Nhãn lớp (class name).
   - Vẽ khung chữ nhật (bounding box) màu đỏ lên đối tượng và viết tên nhãn kèm xác suất lên góc trên của khung.
4. **Ghi Video:** Lưu khung hình đã được vẽ đè thông tin vào video đầu ra bằng `cv2.VideoWriter`.
5. **Giải phóng tài nguyên:** Đóng các kết nối đọc/ghi video sau khi kết thúc.

---

## 5. Hướng dẫn thay đổi video chạy thử nghiệm (Custom Video Input)

Bạn có thể thay đổi video chạy thử nghiệm bất kỳ lúc nào theo hai cách sau:

### Cách 1: Truyền tham số qua dòng lệnh (Khuyên dùng)
Bạn không cần sửa đổi mã nguồn, chỉ cần truyền đường dẫn của video mới thông qua tham số `--video` và chọn đường dẫn lưu kết quả thông qua `--output` khi chạy script:
```powershell
.\venv\Scripts\python src/baseline.py --video "data/input/video_cua_ban.mp4" --output "data/output/video_cua_ban.mp4"
```
*Lưu ý: Bạn có thể thay đổi các tham số lọc độ tin cậy bằng cách truyền thêm `--conf 0.3` (nếu muốn tăng độ chính xác, giảm nhiễu) hoặc thay đổi model khác qua `--model yolov8s.pt`.*

### Cách 2: Thay thế file mặc định
1. Sao chép video mới của bạn vào thư mục `data/input/`.
2. Đổi tên video mới của bạn thành `test.mp4` (thay thế tệp test.mp4 hiện tại).
3. Chạy lệnh mặc định:
   ```powershell
   .\venv\Scripts\python src/baseline.py
   ```
   *Kết quả sẽ tự động lưu vào tệp mặc định `data/output/baseline.mp4`.*

