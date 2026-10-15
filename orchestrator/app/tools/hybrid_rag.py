import re
import math
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

MULTILINGUAL_STOPWORDS = {
    # English functional stopwords
    "to", "and", "or", "the", "is", "in", "at", "of", "a", "an", "for", "on", "with", "by", "from", "as", "be",
    # Marathi / Hindi functional stopwords
    "आणि", "व", "किंवा", "आहे", "नाही", "हे", "ते", "हा", "ही", "की", "का", "मध्ये", "वर", "ने", "ला"
}

def tokenize_multilingual(text: str) -> List[str]:
    """
    Multilingual tokenizer preserving Devanagari Unicode characters, 
    Latin alphanumeric words, and civic identifiers (e.g. '1912', '101', 'ward 53'),
    while filtering out trivial stopwords.
    """
    if not text:
        return []
    # Match contiguous Devanagari characters (\u0900-\u097F) or Latin alphanumeric words
    raw_tokens = re.findall(r"[\u0900-\u097Fa-zA-Z0-9]+", text.lower())
    return [
        t for t in raw_tokens 
        if (len(t) > 1 or t.isdigit()) and t not in MULTILINGUAL_STOPWORDS
    ]


class BM25Scorer:
    """
    Lightweight, zero-dependency BM25 (Okapi) implementation tailored for 
    multilingual municipal knowledge chunks (Marathi, Hindi, English).
    """
    def __init__(self, docs: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.n_docs = len(docs)
        self._build_index()

    def _build_index(self):
        total_len = 0
        for doc in self.docs:
            combined_text = " ".join([
                doc.get("title", ""),
                doc.get("content_mr", ""),
                doc.get("content_en", ""),
                " ".join(doc.get("keywords", []))
            ])
            tokens = tokenize_multilingual(combined_text)
            self.doc_tokens.append(tokens)
            l = len(tokens)
            self.doc_lens.append(l)
            total_len += l

            # Track unique tokens per document for document frequency (DF)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = (total_len / self.n_docs) if self.n_docs > 0 else 1.0

        # Calculate IDF with standard Robertson-Spärck Jones formula
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[int, float]]:
        """
        Score all documents against query using BM25.
        Returns list of (doc_index, score) sorted by score descending.
        """
        q_tokens = tokenize_multilingual(query)
        if not q_tokens or self.n_docs == 0:
            return []

        scores = []
        for idx in range(self.n_docs):
            doc_toks = self.doc_tokens[idx]
            d_len = self.doc_lens[idx]
            doc_score = 0.0

            # Count term frequencies in this document
            tf: Dict[str, int] = {}
            for t in doc_toks:
                tf[t] = tf.get(t, 0) + 1

            for q_term in q_tokens:
                if q_term in tf:
                    term_idf = self.idf.get(q_term, 0.0)
                    term_tf = tf[q_term]
                    numerator = term_tf * (self.k1 + 1.0)
                    denominator = term_tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                    doc_score += term_idf * (numerator / denominator)

            if doc_score > 0.0:
                scores.append((idx, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class HybridMunicipalKnowledgeBase:
    """
    Enterprise Municipal Knowledge Base implementing Hybrid RAG:
    - Sparse BM25 lexical matcher (zero latency, exact Marathi & English keywords).
    - Dense Vector search using OpenAI text-embedding-3-small (1536 dim).
    - Reciprocal Rank Fusion (RRF) to merge and rank results.
    - Resilient fallback to pure in-memory search if vector/API is unavailable.
    """
    def __init__(
        self,
        docs: Optional[List[Dict[str, Any]]] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        from app.tools.rag_knowledge import MUNICIPAL_KNOWLEDGE_DOCS
        self.docs = docs or MUNICIPAL_KNOWLEDGE_DOCS
        self.bm25 = BM25Scorer(self.docs)
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.openai_client = None
        self.doc_embeddings: Dict[int, np.ndarray] = {}

        self._init_openai()
        self._init_doc_embeddings()

    def _init_openai(self):
        if self.api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.api_key)
                logger.info("HybridRAG: OpenAI embedding client initialized.")
            except Exception as e:
                logger.warning(f"HybridRAG: OpenAI client initialization skipped ({e}).")

    def _init_doc_embeddings(self):
        """Pre-compute embeddings for municipal documents if OpenAI client is available."""
        if not self.openai_client:
            return

        try:
            texts_to_embed = []
            for d in self.docs:
                txt = f"{d.get('title', '')} | {d.get('content_mr', '')} | {d.get('content_en', '')}"
                texts_to_embed.append(txt)

            res = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts_to_embed
            )
            for idx, item in enumerate(res.data):
                self.doc_embeddings[idx] = np.array(item.embedding, dtype=np.float32)
            logger.info(f"HybridRAG: Pre-computed vector embeddings for {len(self.doc_embeddings)} documents.")
        except Exception as e:
            logger.warning(f"HybridRAG: Could not pre-compute document embeddings ({e}). Running in BM25 mode.")

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        if not self.openai_client or not query:
            return None
        try:
            res = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query.strip()
            )
            return np.array(res.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.debug(f"HybridRAG query embedding failed: {e}")
            return None

    def search(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Execute Hybrid Search combining BM25 and Vector Cosine Similarity via RRF.
        100% backwards-compatible with MunicipalKnowledgeBase.search().
        """
        if not query or not query.strip():
            return {"found": False, "results": []}

        # 1. Run BM25 Lexical Ranking
        bm25_ranks = self.bm25.score(query)
        bm25_rank_map = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_ranks)}

        # 2. Run Dense Vector Cosine Similarity (if embeddings available)
        q_vec = self._get_query_embedding(query)
        vector_rank_map: Dict[int, int] = {}
        top_vec_sim = 0.0
        if q_vec is not None and self.doc_embeddings:
            vec_scores = []
            for doc_idx, doc_vec in self.doc_embeddings.items():
                denom = (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec))
                if denom > 0:
                    sim = float(np.dot(q_vec, doc_vec) / denom)
                    # Filter out irrelevant noise (minimum cosine similarity 0.38)
                    if sim >= 0.38:
                        vec_scores.append((doc_idx, sim))
            vec_scores.sort(key=lambda x: x[1], reverse=True)
            if vec_scores:
                top_vec_sim = vec_scores[0][1]
            vector_rank_map = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(vec_scores)}

        # 3. Guardrail: If no BM25 matches AND vector similarity is poor (< 0.42), mark as out-of-scope
        has_bm25_match = len(bm25_ranks) > 0
        if not has_bm25_match and top_vec_sim < 0.42:
            return {
                "found": False,
                "message": "या विषयाबद्दल थेट माहिती उपलब्ध नाही, तरी वॉर्ड कार्यालयात किंवा १८००-२३३-००५० वर संपर्क साधावा.",
                "results": []
            }

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF_Score = 1 / (60 + Rank_BM25) + 1 / (60 + Rank_Vector)
        rrf_scores: Dict[int, float] = {}
        all_candidates = set(bm25_rank_map.keys()).union(set(vector_rank_map.keys()))

        if not all_candidates:
            return {
                "found": False,
                "message": "या विषयाबद्दल थेट माहिती उपलब्ध नाही, तरी वॉर्ड कार्यालयात किंवा १८००-२३३-००५० वर संपर्क साधावा.",
                "results": []
            }

        k = 60.0
        for doc_idx in all_candidates:
            score = 0.0
            if doc_idx in bm25_rank_map:
                score += 1.0 / (k + bm25_rank_map[doc_idx])
            if doc_idx in vector_rank_map:
                score += 1.0 / (k + vector_rank_map[doc_idx])
            rrf_scores[doc_idx] = score

        # Sort by RRF score descending
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = [self.docs[idx] for idx, _ in sorted_rrf[:top_k]]

        best_score = sorted_rrf[0][1] if sorted_rrf else 0.0

        return {
            "found": True,
            "results": top_results,
            "confidence": round(best_score * 30.0, 4),
            "method": "hybrid_bm25_vector" if vector_rank_map else "bm25_lexical"
        }
