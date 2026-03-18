import json
from pathlib import Path
from app.rag import RAGService

TEST_CASES = [
    {
        "question": "What are the symptoms of diabetes?",
        "expected_contains": "symptom",
    },
    {
        "question": "How is asthma treated?",
        "expected_contains": "treat",
    },
    {
        "question": "What causes anemia?",
        "expected_contains": "cause",
    },
]


def main():
    rag = RAGService()
    rows = []

    for case in TEST_CASES:
        result = rag.answer(case["question"])
        answer = result["answer"].lower()
        expected = case["expected_contains"].lower()
        row = {
            "question": case["question"],
            "expected_contains": expected,
            "answer": result["answer"],
            "hit_expected": expected in answer,
            "debug": {"sources": result["sources"]},
        }
        rows.append(row)

    output = {"total": len(rows), "passed": sum(r["hit_expected"] for r in rows), "rows": rows}
    Path("./storage").mkdir(exist_ok=True)
    Path("./storage/eval_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
