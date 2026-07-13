# Chunking Pipeline

Pipeline chunking biến nội dung thô của document thành các đoạn văn bản nhỏ, có cấu trúc và đã qua kiểm tra, sẵn sàng để đánh chỉ mục.

```text
RawDocument
    ↓
Normalize
    ↓
Clean
    ↓
Metadata Enrich
    ↓
Section Detect
    ↓
Chunk
    ↓
Chunk Validate
    ↓
list[Chunk]
```

Mỗi bước trong pipeline là một thành phần độc lập, có thể thay thế dễ dàng mà không ảnh hưởng đến các bước khác.

---

## Mục đích

Thay vì chunk trực tiếp từ raw text, pipeline chia nhỏ quá trình thành các giai đoạn rõ ràng. Mỗi giai đoạn chịu trách nhiệm một phần công việc, giúp dễ bảo trì, dễ test và dễ thay đổi thuật toán sau này.

---

## Các giai đoạn

### 1. Normalize

Bước này chuẩn hóa văn bản ở mức ký tự và khoảng trắng:
- Chuẩn hóa Unicode về dạng NFKC.
- Sửa chữa các ký tự xuống dòng khác nhau.
- Thu gọn khoảng trắng dư thừa.

Mục tiêu là đưa văn bản về một dạng thống nhất trước khi xử lý tiếp.

### 2. Clean

Bước này loại bỏ các thành phần không phải nội dung chính:
- Các template dạng `{{...}}` lồng nhau.
- Tham chiếu và thẻ HTML.
- Wikilinks dạng `[[Target|Display]]`.
- Các markup in đậm, in nghiêng.
- Các magic word như `__NOTOC__`.

Mục tiêu là giữ lại chỉ nội dung văn bản thuần túy.

### 3. Metadata Enrich

Bước này bổ sung các metadata suy diễn được từ document:
- Tiêu đề document.
- Nguồn document.
- Các thông tin phục vụ cho việc enrich context sau này.

Trong tương lai, bước này có thể mở rộng thêm: phát hiện ngôn ngữ, loại document, URL nguồn.

### 4. Section Detect

Bước này phát hiện cấu trúc phần mục của document:
- Nhận diện heading dạng Markdown.
- Fallback sang nhận diện heading ngắn, đứng một mình.
- Tách document thành các section có tiêu đề và cấp độ.

Mục tiêu là đảm bảo chunk sau này không vượt qua ranh giới section quan trọng.

### 5. Chunk

Đây là bước tách document thành các đoạn nhỏ. Chunker được thiết kế như một thành phần có thể thay thế.

Hai chunker hiện có:
- **RecursiveChunker**: tách theo đoạn văn, câu, và cuối cùng là từ. Đơn giản, nhanh, nhưng không giữ được cấu trúc.
- **StructureChunker**: giữ các list item chung một chunk, tôn trọng ranh giới section, thêm context prefix cho mỗi chunk, và đánh dấu các section tham khảo.

### 6. Validate

Bước cuối cùng kiểm tra và loại bỏ các chunk không hợp lệ:
- Chunk rỗng.
- Chunk quá ngắn so với ngưỡng min_tokens.
- Chunk quá dài so với ngưỡng max_tokens.

---

## Tính linh hoạt

Mỗi stage trong pipeline đều là một protocol. Điều này cho phép:
- Thay đổi chunker mà không cần sửa pipeline chính.
- Thay thế cleaner cho loại dữ liệu khác nhau.
- Thêm enricher mới để bổ sung metadata.
- Thay đổi section detector cho các định dạng document khác nhau.

---

## Đầu ra

Pipeline trả về danh sách các `Chunk` thuộc tầng storage. Mỗi chunk chứa:
- Tham chiếu đến document cha.
- Thứ tự trong document.
- Nội dung text.
- Số lượng token ước tính.
- Metadata về section path, tiêu đề section, và cờ đánh dấu section tham khảo.

---

## So sánh chunker

| Chunker | Ưu điểm | Nhược điểm |
|---------|---------|------------|
| RecursiveChunker | Đơn giản, nhanh, không phụ thuộc định dạng | Có thể cắt giữa list, không có context section |
| StructureChunker | Giữ list, tôn trọng section, có context, đánh dấu reference | Phức tạp hơn một chút |

Chất lượng của hai chunker nên được đánh giá bằng các metric như Recall@K trên cùng một tập query.
