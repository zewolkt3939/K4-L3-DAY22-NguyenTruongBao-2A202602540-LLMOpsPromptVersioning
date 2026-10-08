# Kết quả lab Day 22 — Nguyễn Trường Bảo

MSSV: **2A202602540**

Repository: https://github.com/zewolkt3939/K4-L3-DAY22-NguyenTruongBao-2A202602540-LLMOpsPromptVersioning

Đã chạy thật với OpenAI `gpt-4o-mini` và LangSmith project `day22-lab`.

- Bước 1: hoàn thành 50 câu hỏi; 50 trace `rag-query`.
- Bước 2: hoàn thành 50 câu hỏi qua A/B routing; 50 trace `ab-rag-query`.
- Hai prompt đã push/pull thành công trong Prompt Hub của tài khoản.
- Bước 3: đánh giá đủ 50 QA/prompt, đủ 4 metric; đạt faithfulness ≥ 0,8 ở cả hai phiên bản.
- Bước 4: chạy 6 mẫu PII giả và 5 mẫu JSON, tạo log thật.
- 6 test offline pass, gồm kiểm tra checkpoint và retry riêng điểm thiếu.
- Lần chạy lại `cd src; python run_all.py` ngày 08/10/2026: cả 4 bước PASS,
  exit code 0; xem `run_all_log.txt`. Bật UTF-8 trên Windows khi chạy.

Project (cần tài khoản có quyền); để xem trực tiếp không đăng nhập, dùng
[100 trace công khai đã xác minh](public_traces.md) (50 RAG + 50 A/B).

URL project để nộp:
https://smith.langchain.com/o/ec85d835-3c23-4c99-9174-e24b327941b4/projects/p/d7ff300a-0b90-4693-b463-98127bb8b2ff

## Kết quả RAGAS

| Metric | V1 | V2 |
|---|---:|---:|
| Faithfulness | 0.9509 | 0.8708 |
| Answer relevancy | 0.9100 | 0.8913 |
| Context recall | 1.0000 | 1.0000 |
| Context precision | 0.9450 | 0.9450 |

V1 có faithfulness cao hơn khoảng 0,0801 và relevancy cao hơn khoảng 0,0187.
Prompt V1 ngắn gọn có thể giúp hạn chế những chi tiết không được context hỗ trợ;
V2 yêu cầu 3–5 câu và mục Supporting details có thể khiến mô hình bổ sung nhiều
mệnh đề hơn. Đây là cách giải thích có thể có, chưa phải kết luận nhân quả.
Trong lần chạy này, nên chọn V1 nếu ưu tiên độ bám tài liệu.

Context của V1/V2 giống nhau trên toàn bộ 50 câu hỏi. Recall bằng nhau;
precision bằng nhau đến 4 chữ số thập phân. Không có bằng chứng prompt
cải thiện retriever. Bộ QA dựa trên knowledge base của lab, chưa đại diện dữ liệu thực tế.

## Bằng chứng và tính minh bạch

- `01_langsmith_traces.png`: ảnh thật giao diện LangSmith lọc hai tên trace, hiển thị 200 traces và error rate 0% sau lần chạy lại.
- `02_prompt_hub.png`: ảnh thật hai prompt trên Hub.
- `02_ab_routing_log.txt`: request ID, phiên bản prompt, câu hỏi và câu trả lời thực tế.
- `03_ragas_scores.png`: biểu đồ tạo trực tiếp từ báo cáo JSON, không phải screenshot dashboard.
- `03_ragas_report.json`: điểm trung bình; `03_ragas_sample_scores.json`: toàn bộ 400 điểm sample/metric.
- `04_pii_demo_log.txt`, `04_json_demo_log.txt`: log chạy Guardrails thật.
- `langsmith_verification.json`: kiểm tra trace và metadata prompt qua API.

`run_all_log.txt` là lần chạy đầy đủ mới nhất: tạo lại 100 câu trả lời,
chấm đủ 400 điểm sample/metric, hoàn tất cả 4 bước trong một lệnh.
JSON điểm chi tiết, báo cáo trung bình và biểu đồ đều lấy từ lần chạy này.
`ragas_resume_log.txt` và `guardrails_run_log.txt` là log lịch sử của lần chạy trước.
Lần trước từng thiếu một điểm context_recall của V1 (câu 11), sau đó phục hồi
thành công. Checkpoint cho phép tiếp tục mà không tạo lại câu trả lời;
chỉ chấm lại điểm thiếu, giữ nguyên mọi điểm thấp hợp lệ.

Evaluator có cảnh báo trả 1 generation thay vì 3 cho answer relevancy; RAGAS
vẫn tính điểm bằng 1 generation. Điều này có thể làm giảm độ ổn định của chỉ số.
Hai prompt hiện có tên `nguyentruongbao-rag-v1` và `nguyentruongbao-rag-v2`.
Lần chạy mới sử dụng hai prompt này; nội dung được dùng chung tại `src/prompts.py`.
Log lịch sử giữ nguyên tên tại thời điểm chạy; ảnh Prompt Hub và xác minh API
được cập nhật theo tên mới.
Project và prompt vẫn riêng tư. Đã chia sẻ riêng 100 trace lab và kiểm tra
toàn bộ liên kết qua API không kèm thông tin xác thực; xem `public_traces.md`.

## Giải thích mã để ôn lab

1. `data_loader.py` chia tài liệu thành đoạn 500 ký tự, chồng 50 ký tự; FAISS lưu vector embedding để tìm các đoạn gần câu hỏi.
2. Bước 1 dùng LCEL: retriever lấy 3 đoạn, ghép context, điền prompt, gọi LLM, chuyển kết quả thành chuỗi. `@traceable` ghi lần hỏi lên LangSmith.
3. `prompts.py` chứa V1 ngắn gọn và V2 có cấu trúc. Bước 2 push/pull qua Hub; MD5 của request ID chẵn/lẻ chọn phiên bản ổn định. Hash dùng cho phân nhóm, không dùng bảo vệ mật khẩu.
4. Bước 3 chạy cùng 50 câu hỏi trên cả hai prompt. `SingleTurnSample` giữ câu hỏi, câu trả lời, danh sách context và đáp án chuẩn. RAGAS chấm bốn chỉ số; không chấp nhận điểm thiếu/NaN.
5. PII validator dùng regex thay thông tin bằng nhãn. `FailResult(fix_value=...)` kết hợp `OnFailAction.FIX` khiến Guard trả đầu ra đã sửa.
6. JSON validator thử parse trước, sau đó gỡ fences, dấu phẩy thừa và thử `ast.literal_eval` cho nháy đơn. Nếu vẫn sai, trả JSON lỗi dự phòng. Đây là sửa cú pháp, chưa kiểm tra schema nghiệp vụ.

Regex PII trong lab không bao phủ mọi định dạng hay mọi quốc gia. Guardrails hiện là demo riêng, chưa tự động bảo vệ các câu trả lời ở bước 1–3.
