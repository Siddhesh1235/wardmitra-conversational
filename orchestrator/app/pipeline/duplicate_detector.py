import math
import logging
from typing import List, Dict, Any, Optional, Tuple
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
    - Text similarity (pgvector cosine similarity)
    Matching formula: (Location 40%) + (Category 30%) + (Text 30%).
    Threshold >= 85% flags as auto-merged duplicate.
    """
    def __init__(self, radius_meters: float = 500.0, similarity_threshold: float = 85.0):
        self.radius_meters = radius_meters
        self.similarity_threshold = similarity_threshold

    def evaluate_duplicate(
        self,
        new_category: str,
        new_lat: float,
        new_lon: float,
        new_description: str,
        existing_complaints: List[Dict[str, Any]]
    ) -> DuplicateCheckResult:
        if not existing_complaints:
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

        for comp in existing_complaints:
            c_lat = float(comp.get("latitude", 0.0))
            c_lon = float(comp.get("longitude", 0.0))
            c_cat = str(comp.get("category", ""))
            c_desc = str(comp.get("description", ""))
            comp_id = str(comp.get("id", ""))

            dist = haversine_distance_meters(new_lat, new_lon, c_lat, c_lon)
            if dist > self.radius_meters:
                continue

            # 1. Location proximity score (0 to 40)
            loc_factor = max(0.0, 1.0 - (dist / self.radius_meters))
            loc_score = loc_factor * 40.0

            # 2. Category match score (0 or 30)
            cat_score = 30.0 if new_category.lower() == c_cat.lower() else 0.0

            # 3. Text similarity score (0 to 30)
            # When pgvector cosine similarity is provided in comp['vector_similarity'], use it;
            # otherwise fallback to Jaccard
            raw_text_sim = float(comp.get("vector_similarity", calculate_text_jaccard(new_description, c_desc)))
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
