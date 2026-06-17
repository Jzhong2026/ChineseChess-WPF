#!/usr/bin/env python3
"""
Demo 9: Local RAG knowledge-base assistant.

Run:
    python Learning/demo9_local_rag/rag_demo.py build
    python Learning/demo9_local_rag/rag_demo.py ask "QSearch solves what problem?"
    python Learning/demo9_local_rag/rag_demo.py search "tool calling"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
SAMPLE_DOCS_DIR = DEMO_DIR / "sample_docs"
DEFAULT_INDEX_PATH = DEMO_DIR / "index.json"
DEFAULT_CONFIG_PATH = DEMO_DIR / "ai_config.json"

SUPPORTED_EXTENSIONS = {".md", ".txt", ".cs", ".py"}
INDEX_VERSION = 1


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str
    term_freq: dict[str, int]
    token_count: int


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def iter_document_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
    return paths


def extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem


def split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n+", text.strip())
    return [re.sub(r"\s+", " ", block).strip() for block in blocks if block.strip()]


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = split_paragraphs(text)
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        if len(current) + 2 + len(paragraph) <= max_chars:
            current = f"{current}\n\n{paragraph}"
            continue

        chunks.append(current)
        overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
        current = f"{overlap}\n\n{paragraph}" if overlap else paragraph

    if current:
        chunks.append(current)

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars * 1.4:
            final_chunks.append(chunk)
            continue
        for start in range(0, len(chunk), max_chars - overlap_chars):
            part = chunk[start : start + max_chars].strip()
            if part:
                final_chunks.append(part)
    return final_chunks


def is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = [char for char in lowered if is_cjk(char)]
    tokens.extend(cjk_chars)
    tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


def build_chunks(docs_root: Path, max_chars: int, overlap_chars: int) -> list[Chunk]:
    paths = iter_document_paths(docs_root)
    if not paths:
        raise FileNotFoundError(f"No supported documents found under: {docs_root}")

    chunks: list[Chunk] = []
    for path in paths:
        text = read_text(path)
        title = extract_title(path, text)
        relative = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        for index, chunk_body in enumerate(chunk_text(text, max_chars, overlap_chars), start=1):
            terms = Counter(tokenize(chunk_body))
            chunks.append(
                Chunk(
                    chunk_id=f"{relative.as_posix()}#{index}",
                    source=relative.as_posix(),
                    title=title,
                    text=chunk_body,
                    term_freq=dict(terms),
                    token_count=sum(terms.values()),
                )
            )
    return chunks


def calculate_idf(chunks: list[Chunk]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(chunk.term_freq.keys())

    total = len(chunks)
    return {
        term: math.log((1 + total) / (1 + count)) + 1.0
        for term, count in document_frequency.items()
    }


def tfidf_vector(term_freq: dict[str, int], idf: dict[str, float]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for term, count in term_freq.items():
        if term in idf:
            vector[term] = (1.0 + math.log(count)) * idf[term]
    return vector


def cosine_score(query: dict[str, float], document: dict[str, float]) -> float:
    numerator = sum(weight * document.get(term, 0.0) for term, weight in query.items())
    query_norm = math.sqrt(sum(weight * weight for weight in query.values()))
    document_norm = math.sqrt(sum(weight * weight for weight in document.values()))
    if query_norm == 0.0 or document_norm == 0.0:
        return 0.0
    return numerator / (query_norm * document_norm)


def save_index(chunks: list[Chunk], idf: dict[str, float], index_path: Path, docs_root: Path) -> None:
    payload = {
        "version": INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "docs_root": str(docs_root),
        "idf": idf,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "text": chunk.text,
                "term_freq": chunk.term_freq,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ],
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_path: Path) -> tuple[list[Chunk], dict[str, float], dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run the build command first."
        )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("version") != INDEX_VERSION:
        raise ValueError("Index version mismatch. Rebuild the index.")

    chunks = [
        Chunk(
            chunk_id=item["chunk_id"],
            source=item["source"],
            title=item["title"],
            text=item["text"],
            term_freq=dict(item["term_freq"]),
            token_count=int(item["token_count"]),
        )
        for item in payload["chunks"]
    ]
    return chunks, dict(payload["idf"]), payload


def retrieve(question: str, chunks: list[Chunk], idf: dict[str, float], top_k: int) -> list[SearchHit]:
    query_terms = Counter(tokenize(question))
    query_vector = tfidf_vector(dict(query_terms), idf)
    exact_terms = {
        term
        for term in re.findall(r"[a-z0-9_]+", question.lower())
        if len(term) >= 2
    }

    hits: list[SearchHit] = []
    for chunk in chunks:
        document_vector = tfidf_vector(chunk.term_freq, idf)
        score = cosine_score(query_vector, document_vector)
        if exact_terms:
            chunk_text = chunk.text.lower()
            exact_hits = sum(1 for term in exact_terms if term in chunk_text)
            score += 0.25 * exact_hits
            title_hits = sum(1 for term in exact_terms if term in chunk.title.lower())
            score += 0.12 * title_hits
        if score > 0:
            hits.append(SearchHit(chunk=chunk, score=score))

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]


def compact(text: str, width: int = 110) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return textwrap.shorten(normalized, width=width, placeholder="...")


def build_prompt(question: str, hits: list[SearchHit]) -> tuple[str, str]:
    system_prompt = (
        "You are a RAG assistant. Answer only from the provided context. "
        "If the context is insufficient, say what is missing. Include source names."
    )
    context_blocks = []
    for number, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[{number}] source={hit.chunk.source} title={hit.chunk.title}\n{hit.chunk.text}"
        )
    user_prompt = f"Question: {question}\n\nContext:\n\n" + "\n\n---\n\n".join(context_blocks)
    return system_prompt, user_prompt


class PlaceholderLlmClient:
    def chat(self, system_prompt: str, user_prompt: str, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return (
                "未检索到足够相关的资料。可以先扩展示例文档，或重新 build 索引后再问。"
            )

        lines = [
            "这是 placeholder 回答，还没有调用真实大模型。",
            "当前先把检索到的证据整理出来，后续配置 ai_config.json 后可切换为 LLM 综合回答。",
            "",
            f"问题：{question}",
            "",
            "相关依据：",
        ]
        for index, hit in enumerate(hits, start=1):
            lines.append(
                f"{index}. {hit.chunk.title} ({hit.chunk.source}, score={hit.score:.3f})"
            )
            lines.append(f"   {compact(hit.chunk.text, width=130)}")
        return "\n".join(lines)


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.api_key = str(config.get("api_key", "")).strip()
        self.base_url = str(config.get("base_url", "")).strip()
        self.model = str(config.get("model", "")).strip()
        self.temperature = float(config.get("temperature", 0.2))
        self.timeout_seconds = int(config.get("timeout_seconds", 60))

    @property
    def is_ready(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def chat(self, system_prompt: str, user_prompt: str, question: str, hits: list[SearchHit]) -> str:
        if not self.is_ready:
            return PlaceholderLlmClient().chat(system_prompt, user_prompt, question, hits)

        request_body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            build_chat_completions_url(self.base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]


def build_chat_completions_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_command(args: argparse.Namespace) -> None:
    chunks = build_chunks(args.docs, args.chunk_size, args.overlap)
    idf = calculate_idf(chunks)
    save_index(chunks, idf, args.index, args.docs)

    print(f"Built index: {args.index}")
    print(f"Documents root: {args.docs}")
    print(f"Chunks: {len(chunks)}")
    print(f"Terms: {len(idf)}")


def search_command(args: argparse.Namespace) -> None:
    question = " ".join(args.question)
    chunks, idf, metadata = load_index(args.index)
    hits = retrieve(question, chunks, idf, args.top_k)

    print(f"Index created at: {metadata.get('created_at')}")
    print(f"Question: {question}")
    print()

    if not hits:
        print("No hits.")
        return

    for index, hit in enumerate(hits, start=1):
        print(f"[{index}] score={hit.score:.3f} source={hit.chunk.source}")
        print(f"    title: {hit.chunk.title}")
        print(f"    text: {compact(hit.chunk.text)}")


def ask_command(args: argparse.Namespace) -> None:
    question = " ".join(args.question)
    chunks, idf, _ = load_index(args.index)
    hits = retrieve(question, chunks, idf, args.top_k)
    system_prompt, user_prompt = build_prompt(question, hits)

    config = load_config(args.config)
    provider = args.provider
    if provider == "auto":
        provider = "openai-compatible" if OpenAICompatibleClient(config).is_ready else "placeholder"

    if provider == "openai-compatible":
        answer = OpenAICompatibleClient(config).chat(system_prompt, user_prompt, question, hits)
    else:
        answer = PlaceholderLlmClient().chat(system_prompt, user_prompt, question, hits)

    print(answer)
    print()
    print("Sources:")
    for index, hit in enumerate(hits, start=1):
        print(f"[{index}] {hit.chunk.source} score={hit.score:.3f}")


def init_config_command(args: argparse.Namespace) -> None:
    if args.config.exists() and not args.force:
        print(f"Config already exists: {args.config}")
        print("Use --force to overwrite it.")
        return

    example = DEMO_DIR / "ai_config.example.json"
    args.config.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created config: {args.config}")
    print("Fill api_key, base_url, and model before using --provider openai-compatible.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG demo with a placeholder LLM client.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a local TF-IDF index.")
    build_parser.add_argument("--docs", type=Path, default=SAMPLE_DOCS_DIR)
    build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    build_parser.add_argument("--chunk-size", type=int, default=900)
    build_parser.add_argument("--overlap", type=int, default=120)
    build_parser.set_defaults(func=build_command)

    search_parser = subparsers.add_parser("search", help="Retrieve source chunks only.")
    search_parser.add_argument("question", nargs="+")
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    search_parser.add_argument("--top-k", type=int, default=4)
    search_parser.set_defaults(func=search_command)

    ask_parser = subparsers.add_parser("ask", help="Retrieve chunks and generate an answer.")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ask_parser.add_argument(
        "--provider",
        choices=["auto", "placeholder", "openai-compatible"],
        default="auto",
    )
    ask_parser.set_defaults(func=ask_command)

    config_parser = subparsers.add_parser("init-config", help="Create ai_config.json from example.")
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_parser.add_argument("--force", action="store_true")
    config_parser.set_defaults(func=init_config_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
