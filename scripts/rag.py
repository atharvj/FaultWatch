"""Local documentation Q&A with pgvector retrieval and evidence-checked generation."""

import argparse
import json
import os
import re
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import psycopg
import torch
from huggingface_hub import snapshot_download
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "README.md",
    "docs/technical-guide.md",
    "docs/dataset.md",
    "docs/model_card.md",
    "docs/benchmark.md",
)
MODELS = {
    "embedding": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "generator": ("google/flan-t5-small", "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"),
}
FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "spiece.model",
)
DEFAULT_DSN = "postgresql://faultwatch:faultwatch-local@127.0.0.1:55435/faultwatch_rag"
MIN_SIMILARITY = 0.27
EVALUATION = (
    (
        "Which dataset is this project using?",
        "docs/dataset.md",
        r"c-mapss|fd001|turbofan engine degradation simulation",
    ),
    ("How many consecutive readings does inference require?", "docs/model_card.md", r"\b20\b"),
    ("What operating condition does FD001 simulate?", "docs/dataset.md", r"sea level"),
    ("Which model is the default for inference?", "docs/model_card.md", r"mlp"),
    ("After how much time do Redis entries expire?", "docs/benchmark.md", r"two minutes"),
    ("What is the population of Japan?", None, None),
    ("How can I renew an expired passport?", None, None),
)


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .")


class Encoder:
    def __init__(self, models: Path):
        root = models / "embedding"
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            root, local_files_only=True, use_safetensors=True
        ).eval()

    def encode(self, texts: list[str]):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.inference_mode():
            output = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)
            return torch.nn.functional.normalize(pooled, dim=1).numpy()


def corpus():
    for source in SOURCES:
        content = re.sub(r"```.*?```", "", (ROOT / source).read_text(), flags=re.S)
        content = re.sub(r"(?<!!)\[([^\]]+)\]\([^)]+\)", r"\1", content)
        content = content.replace("`", "")
        for paragraph in re.split(r"\n\s*\n", content):
            text = " ".join(paragraph.split())
            if not text.startswith(("#", "|", "![", "- ")) and 45 <= len(text) <= 1000:
                yield source, text


def index_documents(connection, encoder: Encoder) -> int:
    documents = list(corpus())
    vectors = encoder.encode([text for _, text in documents])
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS faultwatch_rag_documents (
            id text PRIMARY KEY, source text NOT NULL, passage text NOT NULL, embedding vector(384) NOT NULL
        )
        """
    )
    connection.execute("DELETE FROM faultwatch_rag_documents")
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO faultwatch_rag_documents VALUES (%s, %s, %s, %s)",
            [
                (sha256(f"{source}\n{text}".encode()).hexdigest(), source, text, vector)
                for (source, text), vector in zip(documents, vectors, strict=True)
            ],
        )
    return len(documents)


class Questions:
    def __init__(self, connection, models: Path, encoder: Encoder):
        self.connection = connection
        self.encoder = encoder
        root = models / "generator"
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            root, local_files_only=True, use_safetensors=True
        ).eval()
        register_vector(connection)

    def ask(self, question: str) -> dict:
        vector = self.encoder.encode([question])[0]
        matches = self.connection.execute(
            """
            SELECT source, passage, 1 - (embedding <=> %s) AS similarity
            FROM faultwatch_rag_documents
            ORDER BY embedding <=> %s, id LIMIT 3
            """,
            (vector, vector),
        ).fetchall()
        retrieved = [source for source, _, _ in matches]
        if matches and matches[0][2] >= MIN_SIMILARITY:
            context = "\n\n".join(passage for _, passage, _ in matches)
            prompt = (
                f"{context}\n\nBased on the passages above, answer with a short exact quote."
                f"\n\n{question}"
            )
            inputs = self.tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=False, num_beams=3, max_new_tokens=34
                )
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            quote = normalize(answer)
            for source, passage, _ in matches:
                if quote and re.search(rf"(?<!\w){re.escape(quote)}(?!\w)", normalize(passage)):
                    return {
                        "answer": answer,
                        "source": source,
                        "evidence": passage,
                        "retrieved": retrieved,
                    }
        return {"answer": None, "source": None, "retrieved": retrieved}


def evaluate(questions: Questions) -> dict:
    cases = []
    start = perf_counter()
    for question, source, pattern in EVALUATION:
        result = questions.ask(question)
        cases.append(
            {
                "question": question,
                "expected_source": source,
                "retrieved_source": any(path == source for path in result["retrieved"]),
                "correct": (
                    result["answer"] is None
                    if pattern is None
                    else bool(
                        result["source"] == source
                        and result["answer"]
                        and re.search(pattern, result["answer"], flags=re.IGNORECASE)
                    )
                ),
                **result,
            }
        )
    expected = [case for case in cases if case["expected_source"] is not None]
    unanswerable = [case for case in cases if case["expected_source"] is None]
    return {
        "pgvector_version": questions.connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0],
        "indexed_passages": questions.connection.execute(
            "SELECT count(*) FROM faultwatch_rag_documents"
        ).fetchone()[0],
        "models": MODELS,
        "min_similarity": MIN_SIMILARITY,
        "answerable": len(expected),
        "retrieval_hits_at_3": sum(case["retrieved_source"] for case in expected),
        "correct_answers": sum(case["correct"] for case in expected),
        "abstained": sum(case["correct"] for case in unanswerable),
        "unanswerable": len(unanswerable),
        "elapsed_seconds": round(perf_counter() - start, 3),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("artifacts/rag/models"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch")
    commands.add_parser("index")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    commands.add_parser("evaluate")
    args = parser.parse_args()
    if args.command == "fetch":
        for kind, (repository, revision) in MODELS.items():
            snapshot_download(
                repository,
                revision=revision,
                allow_patterns=FILES,
                local_dir=args.models / kind,
                token=False,
                max_workers=4,
            )
    else:
        torch.set_num_threads(2)
        encoder = Encoder(args.models)
        with psycopg.connect(
            os.environ.get("FAULTWATCH_RAG_DATABASE_URL", DEFAULT_DSN)
        ) as connection:
            if args.command == "index":
                result = {"documents": index_documents(connection, encoder)}
            else:
                questions = Questions(connection, args.models, encoder)
                result = (
                    questions.ask(args.question) if args.command == "ask" else evaluate(questions)
                )
        print(json.dumps(result, indent=2))
        if args.command == "evaluate" and (
            result["correct_answers"] != result["answerable"]
            or result["abstained"] != result["unanswerable"]
        ):
            raise SystemExit(1)
