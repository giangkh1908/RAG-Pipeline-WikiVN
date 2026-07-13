> **Nguồn dữ liệu:** [Vietnam Tourism v2](https://www.kaggle.com/datasets/vuonglsts/vietnam-tourism-v2/data) trên Kaggle.

# Dataset: Vietnam Tourism v2

Bộ dữ liệu Hỏi và Đáp về Du lịch Việt Nam cung cấp cái nhìn toàn diện về ngành du lịch của đất nước, bao gồm nhiều chủ đề như văn hóa, lịch sử, địa điểm nổi tiếng, ẩm thực và các mẹo du lịch thực tế.

---

## Phạm vi nội dung

Các chủ đề chính trong bộ dữ liệu:

- **Văn hóa và Lịch sử:** Các di tích, lễ hội, phong tục tập quán.
- **Địa điểm nổi tiếng:** Từ các thành phố lớn như Hà Nội, TP.HCM đến các kỳ quan thiên nhiên như Vịnh Hạ Long, Sa Pa.
- **Ẩm thực:** Đặc sản các vùng miền, văn hóa ẩm thực đường phố.
- **Mẹo du lịch:** Visa, đi lại, lưu trú, an toàn và sức khỏe.

---

## Cấu trúc dữ liệu gốc

Dữ liệu được cung cấp dưới dạng các tệp JSON, chia thành các tập như train và valid. Mỗi tệp chứa một danh sách các chủ đề, trong đó mỗi chủ đề bao gồm:

- **title:** Tiêu đề chủ đề.
- **paragraphs:** Danh sách các đoạn văn.
  - **context:** Nội dung kiến thức chính.
  - **qas:** Các cặp câu hỏi và câu trả lờiliên quan đến context.

---

## Cách sử dụng trong dự án

Dự án không sử dụng toàn bộ bộ dữ liệu dưới dạng gốc. Thay vào đó, dữ liệu được tách thành hai phần:

### 1. Corpus kiến thức

Từ mỗi chủ đề, chỉ lấy **title** và **context**. Các cặp câu hỏi-câu trả lờitrong `qas` được loại bỏ khỏi corpus.

Mỗi context trở thành một `Document` trong tầng storage. Sau đó, document được đưa qua chunking pipeline để tạo thành các `Chunk` nhỏ hơn, phù hợp để đánh chỉ mục và retrieval.

### 2. Evaluation queries

Các cặp câu hỏi-câu trả lờitrong `qas` được trích xuất thành tập đánh giá. Mỗi eval query bao gồm:

- `id`: Mã định danh của câu hỏi.
- `title`: Chủ đề chứa câu hỏi.
- `question`: Câu hỏi.
- `answer`: Câu trả lờimong đợi.
- `context`: Đoạn văn gốc để tham chiếu.

Eval query được dùng để đo lường chất lượng retrieval sau này. Thay vì dùng toàn bộ hàng nghìn câu hỏi, dự án chỉ chọn một câu hỏi đại diện cho mỗi chủ đề nhằm giảm thờigian đánh giá trong khi vẫn giữ được sự đa dạng.

---

## Tiềm năng ứng dụng

Bộ dữ liệu phù hợp cho:

- Huấn luyện và đánh giá các hệ thống Hỏi và Đáp.
- Phát triển chatbot du lịch thông minh.
- Phân tích và khai thác thông tin du lịch.
- Nghiên cứu xử lý ngôn ngữ tự nhiên tiếng Việt.

---

## Lưu ý về bản quyền và sử dụng

Dữ liệu được lấy từ Kaggle. Ngườidùng cần tuân thủ các điều khoản và giấy phép của tác giả dataset khi sử dụng cho mục đích thương mại hoặc công khai.
