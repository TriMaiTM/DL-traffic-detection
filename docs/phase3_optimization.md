# Tài liệu Phase 3: So sánh tương quan chỉ số & Tối ưu hóa mô hình

Tài liệu này cung cấp kết quả đánh giá thực nghiệm toàn diện và chi tiết trên tập dữ liệu Validation (156 ảnh) đối với cả 3 mô hình thử nghiệm. Phân tích này không chỉ dựa trên `mAP` hay `Loss` mà so sánh tương quan giữa các chỉ số phổ biến: **Precision (Độ chính xác)**, **Recall (Độ nhạy)**, **F1-Score (Điểm F1)**, và **mAP**.

---

## 1. So sánh tương quan tổng thể (Global Metrics Comparison)

Dưới đây là bảng so sánh tương quan các chỉ số trung bình (All classes) của 3 mô hình:

| Mô hình thử nghiệm | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD + No Cosine** (`yolov8_sgd_cos_false`) | **0.838** | 0.840 | 0.839 | 0.899 | 0.601 |
| **Adam + No Cosine** (`yolov8_adam_cos_false`) | 0.742 | 0.805 | 0.772 | 0.843 | 0.558 |
| **SGD + Cosine Decay** (`yolov8_sgd_cos_true`) | 0.826 | **0.862** | **0.843** | **0.921** | **0.617** |

### Nhận xét tổng quan:
*   **SGD + Cosine Decay** đạt sự cân bằng tốt nhất giữa Precision và Recall, dẫn tới **F1-Score đạt 84.3%** và **mAP@0.5 đạt 92.1%** (đều cao nhất trong 3 mô hình).
*   **Recall (Độ nhạy) của SGD + Cosine Decay** tăng vượt trội lên **86.2%** (tăng 2.2% so với SGD thường và 5.7% so với Adam). Điều này có ý nghĩa cực kỳ quan trọng trong bài toán giám sát giao thông vì nó giúp giảm thiểu tối đa việc bỏ sót phương tiện (nhất là xe máy nhỏ hoặc xe khuất bóng).
*   **Adam** cho thấy hiệu năng kém nhất trên tập dữ liệu này. Nó bị bão hòa sớm, cho ra Precision thấp (74.2%) và F1-Score chỉ đạt 77.2%.

---

## 2. So sánh tương quan chi tiết theo từng lớp (Class-wise Metrics Comparison)

### Lớp: CAR (Xe hơi)
| Mô hình | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD + No Cosine** | 0.872 | 0.898 | **0.885** | 0.934 | 0.695 |
| **Adam + No Cosine** | 0.852 | **0.908** | 0.879 | 0.936 | 0.687 |
| **SGD + Cosine Decay** | **0.877** | 0.884 | 0.881 | **0.954** | **0.719** |

*Nhận xét:* Xe hơi là lớp dễ nhận diện nhất. Cả 3 mô hình đều đạt F1-Score trên 87%. Phiên bản SGD + Cosine Decay tối ưu hóa tốt hơn về độ khớp bounding box (mAP@0.5:0.95 đạt **71.9%**).

### Lớp: MOTOR (Xe máy)
| Mô hình | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD + No Cosine** | **0.853** | 0.813 | 0.832 | 0.902 | 0.468 |
| **Adam + No Cosine** | 0.824 | 0.836 | 0.830 | 0.884 | 0.440 |
| **SGD + Cosine Decay** | 0.797 | **0.879** | **0.836** | **0.908** | **0.472** |

*Nhận xét:* Lớp xe máy ở Việt Nam có đặc thù mật độ dày, chồng chéo lên nhau. SGD + Cosine Decay giúp tăng mạnh **Recall lên 87.9%** (giúp phát hiện hầu hết xe máy trên đường), dù Precision giảm nhẹ nhưng tổng thể **F1-Score vẫn cao nhất (83.6%)**.

### Lớp: TRUCK (Xe tải)
| Mô hình | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD + No Cosine** | 0.734 | **0.879** | 0.800 | 0.903 | 0.625 |
| **Adam + No Cosine** | 0.722 | 0.818 | 0.767 | 0.875 | 0.610 |
| **SGD + Cosine Decay** | **0.775** | 0.858 | **0.814** | **0.907** | **0.645** |

*Nhận xét:* SGD + Cosine Decay cải thiện đáng kể **Precision (77.5%)** so với SGD thường (73.4%), tức là giảm thiểu việc dự đoán nhầm các đối tượng khác thành xe tải, đưa F1-Score lên **81.4%**.

### Lớp: BUS (Xe buýt)
| Mô hình | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD + No Cosine** | **0.892** | 0.771 | 0.827 | 0.857 | 0.617 |
| **Adam + No Cosine** | 0.569 | 0.657 | 0.610 | 0.678 | 0.494 |
| **SGD + Cosine Decay** | 0.853 | **0.826** | **0.839** | **0.916** | **0.632** |

*Nhận xét:* Đây là lớp có số lượng mẫu ít nhất trong dataset và dễ gây nhầm lẫn với xe khách hoặc xe tải lớn. 
- **Adam** thất bại nặng nề ở lớp này với F1 chỉ đạt **61.0%** và Precision cực thấp (**56.9%**).
- **SGD + Cosine Decay** vượt trội hoàn toàn với **mAP@0.5 đạt 91.6%** (tăng 5.9% so với SGD thường) và cải thiện mạnh Recall lên **82.6%**, khắc phục triệt để bài toán mất cân bằng dữ liệu lớp hiếm.

---

## 3. Biểu đồ so sánh trực quan

Biểu đồ so sánh đường cong Loss và mAP@0.5 của cả 3 mô hình được xuất tại [comparison_plots.png](file:///d:/HK8/DL/runs/train/comparison_plots.png):

![Biểu đồ so sánh hiệu năng](/d:/HK8/DL/runs/train/comparison_plots.png)

---

## 4. Kết luận lựa chọn mô hình chính thức cho Phase 4

Từ kết quả phân tích tương quan đa chiều:
1. **Lựa chọn:** Trọng số mô hình tốt nhất của lượt chạy **SGD + Cosine Decay** (`runs/train/yolov8_sgd_cos_true/weights/best.pt`) sẽ được sử dụng làm mô hình chính thức.
2. **Lý do lựa chọn:**
   - Đạt **F1-Score cao nhất (84.3%)** và **mAP@0.5 cao nhất (92.1%)**.
   - Khả năng bao phủ, tránh bỏ sót phương tiện cực tốt (**Recall đạt 86.2%**).
   - Thể hiện sự vượt trội rõ rệt nhất ở các lớp khó (như xe máy mật độ dày và xe buýt dữ liệu ít).
