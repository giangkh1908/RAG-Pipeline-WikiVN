# Storage Layer

Tầng lưu trữ mô hình hóa dữ liệu RAG theo hệ thống phân cấp:

```text
Source → Document → Chunk → Index
```

Mọi thực thể đều dùng UUID làm khóa chính.

---

## Mục đích

Tầng storage có nhiệm vụ lưu trữ và quản lý vòng đờicủadữ liệu từ khi vào hệ thống cho đến khi được đánh chỉ mục. Nó tách biệt rõ ràng giữa nội dung thô, nội dung đã xử lý, các đoạn văn bản nhỏ và các vector embedding.

---

## Các thực thể

### Source

Đại diện cho một nguồn dữ liệu bên ngoài. Một source thuộc về một tenant và có một loại cụ thể, ví dụ Wikipedia, văn bản pháp luật, hoặc tài liệu nội bộ.

Các trường chính: id, tenant_id, type, version, metadata.

### Document

Đại diện cho một tài liệu cụ thể thuộc về một source. Document theo dõi trạng thái xử lý và checksum để phát hiện thay đổi nội dung.

Các trường chính: id, source_id, checksum, status, metadata.

Trạng thái của document di chuyển theo luồng: pending → processing → indexed hoặc failed.

### Chunk

Đại diện cho một đoạn văn bản nhỏ được tách ra từ document sau quá trình chunking. Mỗi chunk biết vị trí của nó trong document thông qua chunk_order.

Các trường chính: id, document_id, chunk_order, content, token_count, metadata.

### IndexEntry

Đại diện cho bản ghi chỉ mục vector của một chunk. Bản ghi này lưu dense vector, sparse vector và các metadata liên quan đến việc tìm kiếm.

Các trường chính: chunk_id, dense_vector, sparse_vector, metadata.

---

## Giao diện trừu tượng

Tầng storage được định nghĩa qua một giao diện chung. Mọi triển khai cụ thể đều phải tuân theo giao diện này.

Các nhóm thao tác:
- Quản lý source: tạo, đọc, liệt kê theo tenant.
- Quản lý document: tạo, đọc, liệt kê theo source hoặc trạng thái.
- Quản lý chunk: tạo, đọc, liệt kê theo document.
- Quản lý index: tạo, đọc, liệt kê theo chunk.

---

## Triển khai

### SQLiteStorage

Triển khai persistent storage bằng SQLite. Mọi thực thể Source, Document, Chunk, IndexEntry đều được lưu xuống file `.db`, giúp dữ liệu không mất khi process kết thúc.

Cấu hình đường dẫn database qua `StorageConfig.db_path`. Giá trị mặc định là `data/rag_storage.db`. Để dùng trong test, có thể truyền `":memory:"` nhằm tạo database trong RAM.

Schema bao gồm các bảng:
- `sources`: lưu thông tin nguồn dữ liệu.
- `documents`: lưu tài liệu, checksum và trạng thái xử lý.
- `chunks`: lưu các đoạn văn bản sau chunking.
- `index_entries`: lưu vector embedding và metadata đánh chỉ mục của từng chunk.

Các bảng được đánh index trên `documents(source_id, status)` và `chunks(document_id, chunk_order)` để tối ưu các truy vấn thường gặp trong ingestion và retrieval.

### Các triển khai tương lai

Nếu scale vượt quá khả năng của SQLite, có thể thay thế bằng PostgreSQL hoặc một storage engine khác mà vẫn giữ nguyên giao diện `Storage` protocol.

---

## Nguyên tắc thiết kế

1. UUID cho mọi khóa chính để đảm bảo an toàn trong môi trường phân tán.
2. Chunk là bất biến sau khi được tạo, được tham chiếu bằng UUID.
3. Document có trạng thái rõ ràng để dễ dàng theo dõi và debug luồng xử lý.
4. Tách biệt nội dung và vector: Chunk chứa text, IndexEntry chứa vector.
