# Storage Layer

Tầng lưu trữ của hệ thống RAG được thiết kế để quản lý vòng đời dữ liệu từ khâu nạp thô (ingestion) cho đến khi truy xuất (retrieval) và quản lý trạng thái hội thoại.

## 🏗 Kiến Trúc Phân Cấp

Dữ liệu được mô hình hóa theo hệ thống phân cấp chặt chẽ, đảm bảo tính toàn vẹn và khả năng truy vết:

```text
Source → Document → Chunk → IndexEntry
```

Mọi thực thể trong hệ thống đều sử dụng **UUID** làm khóa chính để đảm bảo tính duy nhất trên quy mô lớn.

---

## 📦 Quản Lý Dữ Liệu RAG (Corpus Storage)

### 1. Source (Nguồn)
Đại diện cho một nguồn dữ liệu bên ngoài (ví dụ: Wikipedia, tập tin PDF, hoặc API dữ liệu du lịch). Một source thuộc về một `tenant` cụ thể.
- **Các trường chính:** `id`, `tenant_id`, `type`, `version`, `metadata`.

### 2. Document (Tài liệu)
Một source có thể chứa nhiều document. Mỗi document theo dõi checksum để phát hiện thay đổi và trạng thái xử lý (`pending` $\rightarrow$ `processing` $\rightarrow$ `indexed`).
- **Các trường chính:** `id`, `source_id`, `checksum`, `status`, `metadata`.

### 3. Chunk (Đoạn văn bản)
Document được chia nhỏ thành các chunk để tối ưu hóa việc tìm kiếm ngữ nghĩa và giới hạn token cho LLM.
- **Các trường chính:** `id`, `document_id`, `chunk_order`, `content`, `token_count`, `metadata`.

### 4. IndexEntry (Chỉ mục Vector)
Lưu trữ đại diện vector của mỗi chunk. Đây là cầu nối giữa tìm kiếm vector và nội dung văn bản.
- **Các trường chính:** `chunk_id`, `dense_vector`, `sparse_vector`, `metadata`.

---

## 💬 Quản Lý Hội Thoại (Conversation Storage)

Để hỗ trợ tính năng chat có bộ nhớ (Memory), hệ thống sử dụng hai bảng bổ sung trong SQLite:

### `chat_sessions`
Theo dõi thông tin phiên làm việc của người dùng.
- **Chức năng:** Lưu thời điểm hoạt động cuối cùng (`last_active_at`), tổng số token đã dùng, và trạng thái nén bộ nhớ (`compacting`).
- **TTL:** Các session không hoạt động quá 24h sẽ được tự động dọn dẹp.

### `chat_turns`
Lưu trữ chi tiết từng lượt tương tác (User $\leftrightarrow$ Assistant).
- **Chức năng:** Lưu câu hỏi, câu trả lời, ý định (`intent`) và bản tóm tắt (`summary`) của lượt đó.
- **Sắp xếp:** Các turn được đánh số thứ tự `turn_no` để tái hiện lịch sử hội thoại chính xác.

---

## 🛠 Triển Khai Kỹ Thuật với SQLite

Hệ thống sử dụng **SQLite** làm database chính cho cả corpus và hội thoại. Để đạt hiệu năng cao và tránh xung đột trong môi trường đa luồng (FastAPI), chúng tôi áp dụng các chiến lược sau:

### 1. Kết Nối Chia Sẻ (Shared Connection)
Thay vì mỗi module (`Storage`, `QueryCache`, `ConversationStore`) mở một kết nối riêng, hệ thống khởi tạo **một kết nối duy nhất** tại `SQLiteStorage` và chia sẻ cho tất cả các thành phần khác. Điều này loại bỏ hoàn toàn lỗi `database is locked`.

### 2. Điều Phối Đa Luồng (Threading Lock)
Sử dụng một `threading.RLock` (Reentrant Lock) chung cho tất cả các thao tác database. Mọi truy vấn SQL đều phải đi qua lock này để đảm bảo tính tuần tự, ngăn chặn xung đột ghi-ghi hoặc đọc-ghi.

### 3. Chế Độ WAL (Write-Ahead Logging)
Kích hoạt `PRAGMA journal_mode=WAL` để cho phép nhiều luồng đọc đồng thời trong khi một luồng khác đang ghi, giúp tăng đáng kể tốc độ phản hồi của API.

### 4. Tối Ưu Truy Vấn
- Đánh index trên `documents(source_id, status)` để tăng tốc độ ingestion.
- Đánh index trên `chunks(document_id, chunk_order)` để truy xuất nhanh nội dung tài liệu.
- Đánh index trên `chat_sessions(last_active_at)` để phục vụ quá trình dọn dẹp (GC) session.

---

## 📐 Nguyên Tắc Thiết Kế
- **Bất Biến (Immutability):** Chunk sau khi tạo không thay đổi nội dung; nếu document thay đổi, toàn bộ chunk tương ứng sẽ được tạo mới.
- **Tách Biệt (Separation):** Tách biệt nội dung văn bản (`Chunk`) và biểu diễn vector (`IndexEntry`) để dễ dàng cập nhật model embedding mà không cần xử lý lại văn bản.
- **Định Danh (Identity):** Luôn dùng UUID thay vì Integer ID để tránh xung đột khi mở rộng sang các database phân tán.
