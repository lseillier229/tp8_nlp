import numpy as np
import re
import tiktoken
import openai
import yaml
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

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
    def __init__(self, chunk_size=256, chunk_overlap=0, embedding_model='BAAI/bge-base-en-v1.5', use_small2big=False, small2big_ratio=2, top_k=5):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_model = embedding_model
        self._use_small2big = use_small2big
        self._small2big_ratio = small2big_ratio
        self._top_k = top_k
        self._embedder = None
        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._chunk_metadata = []  # Store position info for small2big
        self._corpus_embedding = None
        self._client = CLIENT

    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename, encoding="utf-8") as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        
        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        return embedder.encode(questions)

    def _compute_chunks(self, texts):
        all_chunks = []
        for txt in texts:
            if self._use_small2big:
                chunks, metadata = chunk_markdown_with_metadata(
                    txt, 
                    chunk_size=self._chunk_size, 
                    chunk_overlap=self._chunk_overlap,
                    large_chunk_size=self._chunk_size * self._small2big_ratio
                )
                self._chunk_metadata.extend(metadata)
            else:
                chunks = chunk_markdown(txt, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
            all_chunks.extend(chunks)
        return all_chunks

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        return embedder.encode(chunks)

    def get_embedder(self):
        if not self._embedder:
            if self._embedding_model == 'sentence-transformers/all-MiniLM-L6-v2':
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embedding_model)
            elif self._embedding_model == 'intfloat/multilingual-e5-large':
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embedding_model)
            else:
                self._embedder = FlagModel(
                    self._embedding_model,
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

    def _get_context(self, query):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-self._top_k:]
        
        if self._use_small2big and self._chunk_metadata:
            # Return large context for small chunks
            return [self._chunk_metadata[i]['large_chunk'] for i in indexes]
        else:
            return [self._chunks[i] for i in indexes]
    


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


def chunk_markdown(md_text: str, chunk_size: int = 128, chunk_overlap: int = 0) -> list[str]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        header_context = " > ".join(section["headers"])
        if header_context:
            header_context += "\n"
        
        content = section["content"]
        sentences = nltk.sent_tokenize(content)
        
        current_chunk_sentences = []
        current_chunk_tokens = len(tokenizer.encode(header_context))
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sent_tokens = len(tokenizer.encode(sentence))
            
            if current_chunk_sentences and (current_chunk_tokens + sent_tokens > chunk_size):
                chunks.append(header_context + " ".join(current_chunk_sentences))
                
                overlap_tokens = 0
                new_chunk_sentences = []
                back_idx = i - 1
                while back_idx >= 0:
                    prev_sent = sentences[back_idx]
                    prev_tokens = len(tokenizer.encode(prev_sent))
                    if overlap_tokens + prev_tokens > chunk_overlap:
                        break
                    new_chunk_sentences.insert(0, prev_sent)
                    overlap_tokens += prev_tokens
                    back_idx -= 1
                
                current_chunk_sentences = new_chunk_sentences
                current_chunk_tokens = len(tokenizer.encode(header_context)) + overlap_tokens
            
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sent_tokens
            i += 1
            
        if current_chunk_sentences:
            chunks.append(header_context + " ".join(current_chunk_sentences))

    return chunks


def chunk_markdown_with_metadata(md_text: str, chunk_size: int = 128, chunk_overlap: int = 0, large_chunk_size: int = 512) -> tuple[list[str], list[dict]]:
    """Create small chunks for retrieval but keep large chunks for context."""
    parsed_sections = parse_markdown_sections(md_text)
    small_chunks = []
    metadata = []

    for section in parsed_sections:
        header_context = " > ".join(section["headers"])
        if header_context:
            header_context += "\n"
        
        content = section["content"]
        sentences = nltk.sent_tokenize(content)
        
        chunk_indices = []
        current_start = 0
        current_tokens = len(tokenizer.encode(header_context))
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sent_tokens = len(tokenizer.encode(sentence))
            
            if i > current_start and (current_tokens + sent_tokens > chunk_size):
                chunk_indices.append((current_start, i))
                
                overlap_tokens = 0
                back_idx = i - 1
                while back_idx >= current_start:
                    prev_sent = sentences[back_idx]
                    prev_tokens = len(tokenizer.encode(prev_sent))
                    if overlap_tokens + prev_tokens > chunk_overlap:
                        break
                    overlap_tokens += prev_tokens
                    back_idx -= 1
                
                current_start = back_idx + 1
                current_tokens = len(tokenizer.encode(header_context))
                for k in range(current_start, i):
                    current_tokens += len(tokenizer.encode(sentences[k]))
            
            current_tokens += sent_tokens
            i += 1
            
        if i > current_start:
            chunk_indices.append((current_start, i))
            
        for start, end in chunk_indices:
            small_text = header_context + " ".join(sentences[start:end])
            small_chunks.append(small_text)
            
            large_tokens = len(tokenizer.encode(small_text))
            l_start = start
            l_end = end
            
            expanded = True
            while expanded and large_tokens < large_chunk_size:
                expanded = False
                if l_start > 0:
                    prev_sent = sentences[l_start - 1]
                    t = len(tokenizer.encode(prev_sent))
                    if large_tokens + t <= large_chunk_size:
                        l_start -= 1
                        large_tokens += t
                        expanded = True
                
                if l_end < len(sentences):
                    next_sent = sentences[l_end]
                    t = len(tokenizer.encode(next_sent))
                    if large_tokens + t <= large_chunk_size:
                        l_end += 1
                        large_tokens += t
                        expanded = True
            
            large_text = header_context + " ".join(sentences[l_start:l_end])
            
            metadata.append({
                'small_chunk': small_text,
                'large_chunk': large_text,
                'position': start
            })

    return small_chunks, metadata
