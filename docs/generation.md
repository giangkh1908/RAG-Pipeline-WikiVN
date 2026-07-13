# Module Generation

Module này chuyển các chunk đã được retrieve thành câu trả lời tự nhiên,
có trích dẫn và hỗ trợ streaming.

## Tổng quan

Giai đoạn generation là phần cuối cùng của RAG pipeline. Nó nhận các chunk
top-k từ tầng retrieval, xây dựng khối context với các trích dẫn đánh số,
và gọi chat model để sinh câu trả lời bằng tiếng Việt.

## Các thành phần

### CitationContextBuilder

Xây dựng chuỗi context từ các đối tượng `RetrievalResult`. Mỗi chunk được
gán một trích dẫn đánh số `[1]`, `[2]`, ... và tiêu đề chunk (nếu có) được
ghi kèm bên cạnh trích dẫn. Builder cũng trả về một ánh xạ từ nhãn trích
dẫn quay lại kết quả gốc, giúp pipeline sau này xây dựng danh sách nguồn.

Cấu hình giới hạn số lượng chunk được đưa vào và độ dài tối đa của mỗi
chunk.

### LLMAnswerGenerator

Gọi endpoint chat completions của OpenRouter với prompt chứa câu hỏi người
dùng và context đã được trích dẫn. Hỗ trợ hai chế độ:

- `generate(query, context)` trả về toàn bộ đối tượng `GeneratedAnswer`.
- `generate_stream(query, context)` sinh từng token câu trả lời khi chúng
  đến.

Generator sử dụng mô hình được cấu hình trong `RAGConfig` (mặc định là
`openai/gpt-4o-mini`). Chế độ streaming phân tích Server-Sent Events từ
OpenRouter và chỉ sinh nội dung delta.

### RAGPipeline

Điều phối toàn bộ luồng:

1. Tiền xử lý câu hỏi thô bằng `LLMQueryProcessor`.
2. Phát progress event cho bước rewrite.
3. Chạy hybrid retrieval.
4. Phát progress event cho retrieval và xây dựng context.
5. Xây dựng context có trích dẫn.
6. Stream các token câu trả lời từ `LLMAnswerGenerator`.
7. Thu thập các trích dẫn xuất hiện trong câu trả lời cuối cùng và phát
   event `done` kèm nội dung trả lời và danh sách nguồn.

Pipeline cũng cung cấp phương thức đồng bộ `answer()` để tiêu thụ stream và
trả về `AnswerResult`.

## Streaming Events

`RAGPipeline.answer_stream()` sinh các đối tượng `GenerationEvent` với các
loại sự kiện sau:

- `progress`: một bước trung gian của pipeline đã hoàn thành (`rewrite`,
  `retrieval`, `context`, `generation`).
- `token`: một token câu trả lời mới.
- `done`: kết quả cuối cùng bao gồm câu trả lời, context và nguồn.
- `error`: retrieval thất bại hoặc không tìm thấy context liên quan.

Mỗi event có thể được serialize thành JSON để sử dụng trong API hoặc
frontend.

## Trích xuất nguồn

Sau khi generation hoàn tất, pipeline quét câu trả lời để tìm các trích
dẫn dạng `[n]`. Chỉ những trích dẫn xuất hiện trong câu trả lời mới được
đưa vào danh sách nguồn cuối cùng, đảm bảo phản hồi chỉ tham chiếu đến các
chunk mà mô hình thực sự sử dụng.

## Xử lý lỗi

Nếu retrieval không trả về kết quả nào, hoặc context builder không thể xây
dựng context hợp lệ, pipeline sẽ phát event `error` với thông báo dự phòng
thay vì cố gắng trả lời mà không có bằng chứng.
