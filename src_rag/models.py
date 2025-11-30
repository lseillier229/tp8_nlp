import numpy as np
import re
import tiktoken
import openai
import yaml

from FlagEmbedding import FlagModel

CONF = yaml.safe_load(open("config.yml"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")

def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 0,
        top_k: int = 5,
        embed_model: str = "BAAI/bge-base-en-v1.5",
    ):
        print(f"[DEBUG RAG.__init__] chunk_size={chunk_size}, "
              f"chunk_overlap={chunk_overlap}, top_k={top_k}, embed_model={embed_model}")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._embed_model = embed_model

        self._embedder = None
        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._corpus_embedding = None
        self._client = CLIENT
        self._file_texts = []          
        self._file_embeddings = None   

        self._chunks = []              
        self._corpus_embedding = None  

        self._chunk_file_ids = []      

        self._loaded_files = set()

    def load_files(self, filenames):
        new_file_texts = []

        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename, encoding="utf-8") as f:
                text = f.read()

            self._loaded_files.add(filename)

            file_id = len(self._file_texts)
            self._file_texts.append(text)
            new_file_texts.append(text)

            film_chunks = chunk_markdown(
                text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

            self._chunks.extend(film_chunks)
            self._chunk_file_ids.extend([file_id] * len(film_chunks))

        if self._file_texts:
            self._file_embeddings = self.embed_corpus(self._file_texts)
        if self._chunks:
            self._corpus_embedding = self.embed_corpus(self._chunks)

    def _normalize_embeddings(self, embs):
        embs = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        return embs / norms


    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        return self.embed_texts(questions)

    def _compute_chunks(self, texts):
        return sum(
            (
                chunk_markdown(
                    txt,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                )
                for txt in texts
            ),
            [],
        )
    def embed_texts(self, texts):
        embedder = self.get_embedder()
        embs = embedder.encode(texts)
        return self._normalize_embeddings(embs)
    def embed_corpus(self, chunks):
        return self.embed_texts(chunks)

    def get_embedder(self):
        if not self._embedder:
            self._embedder = FlagModel(
                self._embed_model,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )

        return self._embedder

    def reply(self, query):
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content
        

    def _build_prompt(self, query):
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply \"I cannot answer that question\".
Query: {query}
Answer:"""

    def _get_context(self, query, n_files: int = 3):
        if self._corpus_embedding is None or self._file_embeddings is None:
            raise ValueError("Corpus or file embeddings not computed. Call load_files() first.")

        q_emb = self.embed_questions([query])   # (1, d)

        # --- Niveau 1 : retrouver les meilleurs films ---
        file_scores = q_emb @ self._file_embeddings.T   # (1, F)
        n_files = min(n_files, self._file_embeddings.shape[0])
        best_file_idx = np.argsort(file_scores[0])[-n_files:]  # indices de films pertinents

        # --- Niveau 2 : chunks restreints à ces films ---
        chunk_file_ids = np.array(self._chunk_file_ids)
        mask = np.isin(chunk_file_ids, best_file_idx)
        candidate_indices = np.where(mask)[0]

        if candidate_indices.size == 0:
            # fallback : comportement ancien (sur tout le corpus)
            sim_scores = q_emb @ self._corpus_embedding.T
            top_k = min(self._top_k, self._corpus_embedding.shape[0])
            idx = np.argsort(sim_scores[0])[-top_k:][::-1]
            return [self._chunks[i] for i in idx]

        candidate_embs = self._corpus_embedding[candidate_indices]  # (C', d)
        sim_scores = q_emb @ candidate_embs.T

        top_k = min(self._top_k, candidate_embs.shape[0])
        local_idx = np.argsort(sim_scores[0])[-top_k:][::-1]
        global_idx = candidate_indices[local_idx]

        return [self._chunks[i] for i in global_idx]

    


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    """
    Parses markdown into a list of {'headers': [...], 'content': ...}
    Preserves full header hierarchy (e.g. ["Section", "Sub", "SubSub", ...])
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # Save previous section
            if current_section["content"]:
                sections.append(current_section)

            # Adjust the header stack
            header_stack = header_stack[:level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(
    md_text: str,
    chunk_size: int = 128,
    chunk_overlap: int = 0,
) -> list[str]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        header = section.get("header", "")
        content = section["content"]

        tokens = tokenizer.encode(content)
        if not tokens:
            continue

        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size

        for i in range(0, len(tokens), step):
            token_chunk = tokens[i:i + chunk_size]
            if not token_chunk:
                continue

            chunk_text = tokenizer.decode(token_chunk)

            if header:
                full_chunk = f"{header}\n\n{chunk_text}"
            else:
                full_chunk = chunk_text

            chunks.append(full_chunk)

    return chunks


