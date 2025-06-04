Đề tài này tập trung vào việc xây dựng một hệ thống nhận diện 10 loại hoa quả khác nhau bằng cách sử dụng mô hình YOLO (You Only Look Once) – một trong những thuật toán nhận diện đối tượng hiệu quả và nhanh chóng nhất hiện nay.
Quá trình thực hiện bao gồm thu thập dữ liệu hình ảnh các loại hoa quả, gán nhãn dữ liệu, và huấn luyện mô hình YOLO trên tập dữ liệu đó. Mô hình sau khi được huấn luyện có khả năng phát hiện và phân loại chính xác các loại hoa quả trong hình ảnh mới với tốc độ nhanh, phù hợp cho các ứng dụng trong lĩnh vực nông nghiệp thông minh, quản lý kho hàng, hoặc các hệ thống bán hàng tự động.
Kết quả thử nghiệm cho thấy mô hình đạt độ chính xác cao trong việc nhận diện và phân loại 10 loại hoa quả, đồng thời thể hiện hiệu suất xử lý thời gian thực, mở ra nhiều hướng phát triển ứng dụng trong thực tế. Đề tài góp phần nâng cao khả năng ứng dụng công nghệ AI trong nông nghiệp và các lĩnh vực liên quan.
Mục tiêu đề tài
·  Xây dựng mô hình học sâu có khả năng nhận diện và phân loại 10 loại hoa quả.
·  Ứng dụng thuật toán YOLO để huấn luyện và đánh giá hiệu suất mô hình.
·  Tạo tập dữ liệu, gán nhãn và huấn luyện mô hình với độ chính xác cao.
·  Đề xuất ứng dụng mô hình vào thực tiễn (ví dụ: hệ thống phân loại hoa quả tự động, kiểm tra chất lượng…).
Phạm vi
·  Đối tượng nhận diện: 10 loại hoa quả cụ thể (ví dụ: táo, chuối, cam, xoài, nho, dứa, dưa hấu, lựu, đu đủ, chanh).
·  Phạm vi kỹ thuật: Áp dụng thuật toán YOLOv5 hoặc YOLOv8 trên tập dữ liệu tùy chỉnh.
·  Không bao gồm: nhận diện độ chín, chất lượng, phân tích hình thái sâu.
Phương pháp thực hiện
Bước 1: Thu thập dữ liệu
Tìm kiếm và tải ảnh từ các nguồn (Kaggle, Google Image, Roboflow).
Lọc dữ liệu đảm bảo chất lượng (độ phân giải, rõ nét, không trùng lặp).
Bước 2: Gán nhãn dữ liệu
Sử dụng phần mềm như LabelImg, Roboflow hoặc CVAT để gán nhãn bounding boxes cho mỗi loại hoa quả.
Bước 3: Huấn luyện mô hình YOLO
Cài đặt môi trường (Python, PyTorch, YOLOv5 repo).
Chia dữ liệu thành tập huấn luyện và tập kiểm thử (train/test).
Thiết lập tham số huấn luyện: epochs, batch size, learning rate, optimizer…
Theo dõi tiến trình huấn luyện thông qua các chỉ số như loss, mAP, precision, recall.
Bước 4: Kiểm thử và đánh giá
Chạy mô hình trên tập kiểm thử để đánh giá độ chính xác.
Vẽ bounding boxes và nhãn dự đoán lên ảnh để kiểm tra kết quả trực quan.
III.Kết quả YoLo huấn luyện




















