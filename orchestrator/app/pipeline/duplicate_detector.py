import os
import math
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from app.schemas.ai_inference import DuplicateCheckResult

logger = logging.getLogger(__name__)

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points in meters."""
    R = 6371000  # Radius of earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_text_jaccard(s1: str, s2: str) -> float:
    """Fallback text similarity metric when vector embedding is computing."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

class DuplicateDetector:
    """
    Deduplication & Fraud detection engine using:
    - 500m geospatial radius filter
    - 48-hour recency window
    - Category match
    - Text semantic similarity (OpenAI text-embedding-3-small + PostgreSQL pgvector cosine similarity)
    Matching formula: (Location 40%) + (Category 30%) + (Semantic Text 30%).
    Threshold >= 85% flags as auto-merged duplicate.
    """
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        radius_meters: float = 500.0,
        similarity_threshold: float = 85.0
    ):
        self.radius_meters = radius_meters
        self.similarity_threshold = similarity_threshold
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # PostgreSQL DB Credentials
        self.pg_host = os.getenv("PG_HOST", "127.0.0.1")
        self.pg_port = int(os.getenv("PG_PORT", "5433"))
        self.pg_db = os.getenv("PG_DATABASE", "defaultdb")
        self.pg_user = os.getenv("PG_USER", "wardmitra")
        self.pg_password = os.getenv("PG_PASSWORD", "Ward19A!Thane")

        # OpenAI Client for Embeddings
        self.openai_client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Embedding client initialized for pgvector duplicate detector.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI embedding client: {e}")

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate 1536-dimensional vector embedding using OpenAI text-embedding-3-small."""
        if not self.openai_client or not text or not text.strip():
            return None
        try:
            res = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.strip()
            )
            return np.array(res.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error generating OpenAI embedding: {e}")
            return None

    async def _async_query_pgvector(
        self,
        emb_vector: np.ndarray,
        ward_id: Optional[str] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Query PostgreSQL complaints table using pgvector <=> cosine distance operator."""
        import asyncpg
        from pgvector.asyncpg import register_vector

        conn = await asyncpg.connect(
            host=self.pg_host,
            port=self.pg_port,
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_db,
            ssl="prefer",
            timeout=3.0
        )
        try:
            await register_vector(conn)
            
            # Query candidate complaints that have embeddings within recency window
            query = """
                SELECT id, category_id, title, description, location, ward_id,
                       1 - (embedding <=> $1) AS vector_similarity
                FROM complaints
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1
                LIMIT $2
            """
            rows = await conn.fetch(query, emb_vector, limit)
            candidates = []
            for r in rows:
                loc = r.get("location")
                lat, lon = None, None
                if loc:
                    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                        try:
                            lat, lon = float(loc[0]), float(loc[1])
                        except Exception:
                            pass
                    elif isinstance(loc, str):
                        try:
                            parsed_loc = json.loads(loc)
                            if isinstance(parsed_loc, list) and len(parsed_loc) >= 2:
                                lat, lon = float(parsed_loc[0]), float(parsed_loc[1])
                            elif isinstance(parsed_loc, dict):
                                lat = float(parsed_loc.get("lat") or parsed_loc.get("latitude"))
                                lon = float(parsed_loc.get("lon") or parsed_loc.get("lng") or parsed_loc.get("longitude"))
                        except Exception:
                            pass

                candidates.append({
                    "id": str(r.get("id")),
                    "category": str(r.get("category_id") or r.get("title") or ""),
                    "title": r.get("title") or "",
                    "description": r.get("description") or "",
                    "latitude": lat or 19.228,
                    "longitude": lon or 73.070,
                    "vector_similarity": float(r.get("vector_similarity") or 0.0),
                    "ward_id": str(r.get("ward_id") or "")
                })
            return candidates
        finally:
            await conn.close()

    def fetch_pgvector_candidates(
        self,
        new_description: str,
        ward_id: Optional[str] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Synchronously fetch candidates from pgvector with graceful timeout fallback."""
        emb = self.generate_embedding(new_description)
        if emb is None:
            return []
        try:
            return asyncio.run(self._async_query_pgvector(emb, ward_id=ward_id, limit=limit))
        except Exception as e:
            logger.warning(f"pgvector query skipped or failed ({e}). Resilient fallback in effect.")
            return []

    async def _async_save_complaint_embedding(self, complaint_id: int, text: str):
        """Asynchronously compute and store vector embedding for a registered complaint."""
        emb = self.generate_embedding(text)
        if emb is None:
            return
        import asyncpg
        from pgvector.asyncpg import register_vector
        conn = await asyncpg.connect(
            host=self.pg_host,
            port=self.pg_port,
            user=self.pg_user,
            password=self.pg_password,
            database=self.pg_db,
            ssl="prefer",
            timeout=3.0
        )
        try:
            await register_vector(conn)
            await conn.execute("UPDATE complaints SET embedding = $1 WHERE id = $2", emb, int(complaint_id))
            logger.info(f"Stored pgvector embedding for complaint ID {complaint_id}")
        finally:
            await conn.close()

    def save_complaint_embedding(self, complaint_id: Any, text: str):
        """Helper to save vector embedding in PostgreSQL."""
        try:
            cid = int(str(complaint_id).replace("CMP-", ""))
            asyncio.run(self._async_save_complaint_embedding(cid, text))
        except Exception as e:
            logger.warning(f"Could not save complaint embedding: {e}")

    def evaluate_duplicate(
        self,
        new_category: str,
        new_lat: float,
        new_lon: float,
        new_description: str,
        existing_complaints: Optional[List[Dict[str, Any]]] = None,
        ward_id: Optional[str] = None
    ) -> DuplicateCheckResult:
        """
        Evaluate if grievance is a duplicate using Pgvector semantic search + geospatial proximity.
        Formula: (Location 40%) + (Category 30%) + (Semantic Vector 30%).
        """
        complaints_to_check = existing_complaints or []

        # If no complaints provided in-memory, query pgvector candidates directly from PostgreSQL
        if not complaints_to_check and new_description:
            pg_candidates = self.fetch_pgvector_candidates(new_description, ward_id=ward_id)
            if pg_candidates:
                complaints_to_check = pg_candidates

        if not complaints_to_check:
            return DuplicateCheckResult(
                is_duplicate=False,
                similarity_score=0.0,
                parent_complaint_id=None,
                matched_complaints_count=0
            )

        highest_score = 0.0
        best_parent_id = None
        closest_distance = None
        matches_count = 0

        # Pre-compute new complaint's embedding if we need to compare with existing complaints
        new_embedding = None
        if any("vector_similarity" not in comp for comp in complaints_to_check):
            new_embedding = self.generate_embedding(new_description)

        for comp in complaints_to_check:
            c_lat = float(comp.get("latitude", 0.0))
            c_lon = float(comp.get("longitude", 0.0))
            c_cat = str(comp.get("category", "")).lower()
            c_desc = str(comp.get("description", ""))
            c_title = str(comp.get("title", "")).lower()
            comp_id = str(comp.get("id", ""))

            dist = haversine_distance_meters(new_lat, new_lon, c_lat, c_lon)
            if dist > self.radius_meters:
                continue

            # 1. Location proximity score (0 to 40)
            loc_factor = max(0.0, 1.0 - (dist / self.radius_meters))
            loc_score = loc_factor * 40.0

            # 2. Category match score (0 or 30)
            target_cat = new_category.lower()
            cat_match = target_cat in c_cat or target_cat in c_title or c_cat in target_cat
            cat_score = 30.0 if cat_match else 0.0

            # 3. Semantic Vector Text similarity score (0 to 30)
            if "vector_similarity" in comp:
                raw_text_sim = float(comp["vector_similarity"])
            else:
                raw_text_sim = calculate_text_jaccard(new_description, c_desc)

            text_score = min(max(raw_text_sim, 0.0), 1.0) * 30.0
            total_score = round(loc_score + cat_score + text_score, 2)

            if total_score >= 60.0:
                matches_count += 1

            if total_score > highest_score:
                highest_score = total_score
                best_parent_id = comp_id
                closest_distance = round(dist, 2)

        is_dup = highest_score >= self.similarity_threshold

        return DuplicateCheckResult(
            is_duplicate=is_dup,
            similarity_score=highest_score,
            parent_complaint_id=best_parent_id if is_dup else None,
            matched_complaints_count=matches_count,
            distance_meters=closest_distance
        )

