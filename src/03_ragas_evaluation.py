"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


from prompts import SYSTEM_V1, SYSTEM_V2, PROMPT_V1, PROMPT_V2, PROMPTS

# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]   # phải là list[str] !

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],        # out["answer"]
            "contexts":  out["contexts"],        # out["contexts"] — phải là list[str] !
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],           # r["question"]
            response=r["answer"],             # r["answer"]
            retrieved_contexts=r["contexts"],   # r["contexts"]
            reference=r["reference"],            # r["reference"]
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    # Gợi ý:
    #   result = evaluate(
    #       dataset,
    #       metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    #       llm=llm_eval,
    #       embeddings=emb_eval,
    #   )
    checkpoint = Path(__file__).parent.parent / "data" / f"ragas_{version}_checkpoint.json"
    signature = hashlib.sha256(json.dumps({"samples": rag_results,
        "provider": config.PROVIDER, "llm": llm_eval.model_dump(),
        "embeddings": emb_eval.model_dump()}, sort_keys=True, default=str).encode()).hexdigest()
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    saved = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    if saved.get("signature") == signature:
        rows = saved["scores"]
        print(f"Resuming saved evaluation for {version}")
    else:
        result = evaluate(dataset, metrics=metrics, llm=llm_eval, embeddings=emb_eval)
        rows = [{metric.name: (float(result[metric.name][i])
                 if result[metric.name][i] is not None and np.isfinite(result[metric.name][i]) else None)
                 for metric in metrics} for i in range(len(rag_results))]

    def save():
        checkpoint.write_text(json.dumps({"signature": signature, "scores": rows},
                                        indent=2, allow_nan=False), encoding="utf-8")
    save()
    # Retry only missing metric/sample pairs; never replace a valid low score.
    for metric in metrics:
        for i, row in enumerate(rows):
            if row.get(metric.name) is not None:
                continue
            for attempt in range(2):
                print(f"Retry {version} sample {i+1}, metric {metric.name}, attempt {attempt+1}")
                retry = evaluate(build_ragas_dataset([rag_results[i]]), metrics=[metric],
                                 llm=llm_eval, embeddings=emb_eval)
                score = retry[metric.name][0]
                if score is not None and np.isfinite(score):
                    row[metric.name] = float(score)
                    save()
                    break
    result = {metric.name: [row.get(metric.name) for row in rows] for metric in metrics}

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        valid = [v for v in raw if v is not None and np.isfinite(v)]
        if not valid or len(valid) != len(raw):
            raise ValueError(f"Incomplete scores for {key}")
        scores[key] = float(np.mean(valid))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main(resume=False):
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    output_path = Path(__file__).parent.parent / "data" / "rag_outputs.json"
    if resume:
        stored = json.loads(output_path.read_text(encoding="utf-8"))
        if stored.get("prompts") != {"v1": SYSTEM_V1, "v2": SYSTEM_V2}:
            raise ValueError("Saved output prompts do not match current prompts")
        v1_results, v2_results = stored["v1"], stored["v2"]
        expected = [(qa["question"], qa["reference"]) for qa in QA_PAIRS]
        for results in [v1_results, v2_results]:
            if [(r["question"], r["reference"]) for r in results] != expected:
                raise ValueError("Saved output QA pairs do not match current dataset")
        print("Resuming saved RAG outputs (no answer regeneration)")
    else:
        vectorstore = setup_vectorstore()
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")
        output_path.write_text(json.dumps({"v1": v1_results, "v2": v2_results,
            "prompts": {"v1": SYSTEM_V1, "v2": SYSTEM_V2}}, indent=2, ensure_ascii=False), encoding="utf-8")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner = "Tie" if s1 == s2 else ("← V1" if s1 > s2 else "← V2")
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    evidence = Path(__file__).parent.parent / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "03_ragas_report.json").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reuse saved answers and metric checkpoints")
    main(resume=parser.parse_args().resume)
