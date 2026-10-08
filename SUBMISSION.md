# Hướng dẫn nộp bài — Day 22: LangSmith + Prompt Versioning

**Nguyễn Trường Bảo — MSSV 2A202602540**

Repo của bài: `K4-L3-DAY22-NguyenTruongBao-2A202602540-LLMOpsPromptVersioning`

Hai đường dẫn để nộp trên VLearn:

1. GitHub: https://github.com/zewolkt3939/K4-L3-DAY22-NguyenTruongBao-2A202602540-LLMOpsPromptVersioning
2. LangSmith project: https://smith.langchain.com/o/ec85d835-3c23-4c99-9174-e24b327941b4/projects/p/d7ff300a-0b90-4693-b463-98127bb8b2ff

Project cần tài khoản có quyền. Minh chứng xem không cần đăng nhập:
[100 trace LangSmith công khai](evidence/public_traces.md).
Học viên tự thực hiện thao tác nộp trên VLearn.

> **Hình thức: Bài CÁ NHÂN.** Mỗi học viên tự nộp 1 repo của riêng mình.

---

## 1. Đặt tên repo

Tạo repo **public** trên GitHub cá nhân với tên theo cấu trúc:

```
K4-L3-DAY22-HoVaTen-MSSV-LLMOpsPromptVersioning
```

Ví dụ: `K4-L3-DAY22-NguyenDongHung-2A20260000-LLMOpsPromptVersioning`

- Viết **không dấu, không khoảng trắng**, các phần ngăn cách bằng dấu `-`.
- Ngày học dùng hai chữ số: `DAY22`.

---

## 2. Cấu trúc repo khi nộp

```
K4-L3-DAY22-HoVaTen-MSSV-LLMOpsPromptVersioning/
├── src/
│   ├── 01_langsmith_rag_pipeline.py   ← đã hoàn thành các TODO
│   ├── 02_prompt_hub_ab_routing.py    ← đã đổi PROMPT_V1_NAME / PROMPT_V2_NAME theo tên bạn
│   ├── 03_ragas_evaluation.py
│   ├── 04_guardrails_validator.py
│   ├── config.py, qa_pairs.py, run_all.py, utils/
├── data/
│   └── knowledge_base.txt
├── evidence/                          ← BẮT BUỘC đủ 7 tệp
│   ├── 01_langsmith_traces.png        (≥ 50 traces)
│   ├── 02_prompt_hub.png              (2 prompt được đặt tên)
│   ├── 02_ab_routing_log.txt          (có nhãn v1/v2)
│   ├── 03_ragas_scores.png            (bảng so sánh V1 vs V2)
│   ├── 03_ragas_report.json           (bản sao data/ragas_report.json)
│   ├── 04_pii_demo_log.txt            (≥ 5 test case)
│   ├── 04_json_demo_log.txt           (≥ 4 test case)
│   └── README.md                      (tùy chọn — phân tích V1 vs V2, có điểm thưởng)
├── .env.example                       ← KHÔNG commit .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Nơi nộp

Nộp qua **cổng nộp bài của khóa học** (LMS) 2 đường dẫn:

1. URL GitHub repository (public, đặt tên đúng quy ước).
2. URL LangSmith project (tổng cộng ≥ 100 traces; để chế độ chia sẻ công khai nếu muốn nhận điểm thưởng).

---

## 4. Deadline

**23:59 ngày 08/10/2026 (giờ Việt Nam, GMT+7)** — tức 23:59 ngày học lab.

Nếu key coach thông báo deadline khác trong vòng 48 giờ sau buổi lab thì theo thông báo đó. Nộp muộn bị trừ điểm theo [RULES.md](RULES.md).

Hệ thống lấy **commit cuối cùng trước deadline** để chấm.

---

## 5. Kiểm tra trước khi nộp

```bash
# 1. .env không bị track
git ls-files | grep -E '^\.env$' && echo "XOÁ .env KHỎI GIT NGAY" || echo "OK: .env không bị commit"

# 2. Không có API key trong mã nguồn
git grep -nE 'sk-[A-Za-z0-9_-]{10,}|lsv2_[A-Za-z0-9_]{10,}|AIza[0-9A-Za-z_-]{20,}' -- . ':!.env.example' || echo "OK: không thấy key"

# 3. Đủ 7 tệp evidence
for f in 01_langsmith_traces.png 02_prompt_hub.png 02_ab_routing_log.txt 03_ragas_scores.png \
         03_ragas_report.json 04_pii_demo_log.txt 04_json_demo_log.txt; do
  [ -s "evidence/$f" ] && echo "OK  $f" || echo "THIẾU $f"
done

# 4. Report RAGAS là JSON hợp lệ
python -m json.tool evidence/03_ragas_report.json > /dev/null && echo "OK: JSON hợp lệ"
```

Sau khi push, mở repo bằng **cửa sổ ẩn danh** để chắc chắn repo public và thấy đủ file.

Nếu `python` báo `UnicodeEncodeError` khi lưu log trên Windows: xem Checkpoint 0 trong [CHECKPOINTS.md](CHECKPOINTS.md) (bật `PYTHONUTF8=1`).
