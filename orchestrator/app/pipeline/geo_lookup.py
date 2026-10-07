import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from app.schemas.ai_inference import GeoLookupResult

logger = logging.getLogger(__name__)

# Primary GeoJSON path (self-contained within orchestrator)
GEOJSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "kalyan.geojson"
if not GEOJSON_PATH.exists():
    GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "kalyan.geojson"

def point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    """Ray-casting algorithm to test if (x, y) is inside polygon coordinates."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class GeoLookupService:
    """
    Municipal Ward Reverse Geocoding Service.
    Determines official Ward ID and Ward Name from GPS coordinates (latitude, longitude).
    """
    def __init__(self, geojson_path: Optional[Path] = None):
        path = geojson_path or GEOJSON_PATH
        self.features = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.features = data.get("features", [])
                logger.info(f"Loaded {len(self.features)} ward boundaries from {path}")
            except Exception as e:
                logger.error(f"Failed to load GeoJSON wards: {e}")
        else:
            logger.warning(f"GeoJSON file not found at {path}")

    def lookup_ward(self, latitude: float, longitude: float) -> GeoLookupResult:
        """
        Reverse geocode GPS coordinates to municipal ward.
        longitude = x, latitude = y
        """
        for feature in self.features:
            geom = feature.get("geometry", {})
            g_type = geom.get("type")
            coords = geom.get("coordinates", [])
            props = feature.get("properties", {})

            ward_code = str(props.get("sourcewardcode") or props.get("objectid") or "Unknown")
            ward_name = props.get("ward_lgd_name") or f"Ward No. {ward_code}"

            if g_type == "Polygon":
                for ring in coords:
                    if point_in_polygon(longitude, latitude, ring):
                        return GeoLookupResult(
                            latitude=latitude,
                            longitude=longitude,
                            ward_id=ward_code,
                            ward_name=ward_name,
                            is_within_boundary=True
                        )
            elif g_type == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        if point_in_polygon(longitude, latitude, ring):
                            return GeoLookupResult(
                                latitude=latitude,
                                longitude=longitude,
                                ward_id=ward_code,
                                ward_name=ward_name,
                                is_within_boundary=True
                            )

        # Fallback if coordinates are slightly outside mapped polygon
        return GeoLookupResult(
            latitude=latitude,
            longitude=longitude,
            ward_id="53",
            ward_name="Kalyan-Dombivli Default Central Ward",
            is_within_boundary=False
        )
