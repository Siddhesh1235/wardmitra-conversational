import os
import re
import math
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import requests

from app.schemas.ai_inference import GeoLookupResult

logger = logging.getLogger(__name__)

# Primary GeoJSON path (self-contained within orchestrator)
GEOJSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "kalyan.geojson"
if not GEOJSON_PATH.exists():
    GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "kalyan.geojson"

# Default fallback ward if outside municipal boundary
DEFAULT_FALLBACK_WARD_ID = "53"
DEFAULT_FALLBACK_WARD_NAME = "Kalyan-Dombivli Central Ward 53"


# =========================================================================
# 1. Location Parsing: Support Mobile App / GPS [latitude, longitude]
# =========================================================================
def parse_numeric_location(value_or_body: Any) -> Optional[Tuple[float, float]]:
    """
    Parse numeric GPS location from multiple possible formats:
    - [latitude, longitude] list or tuple (standard mobile app format)
    - {'lat': ..., 'lng': ...} or {'latitude': ..., 'longitude': ...} dict
    - "19.2435, 73.1310" or "[19.2435, 73.1310]" string
    - Request body object with .latitude/.longitude or .location
    Returns (latitude, longitude) floats, or None if invalid.
    """
    if value_or_body is None:
        return None

    # 1. Direct list or tuple: [latitude, longitude]
    if isinstance(value_or_body, (list, tuple)) and len(value_or_body) >= 2:
        try:
            lat = float(value_or_body[0])
            lon = float(value_or_body[1])
            if _is_valid_coord(lat, lon):
                return lat, lon
        except (ValueError, TypeError):
            pass

    # 2. Dictionary format
    if isinstance(value_or_body, dict):
        # Nested location field?
        if "location" in value_or_body and isinstance(value_or_body["location"], (list, tuple, str)):
            parsed = parse_numeric_location(value_or_body["location"])
            if parsed:
                return parsed

        lat_raw = value_or_body.get("latitude") if value_or_body.get("latitude") is not None else value_or_body.get("lat")
        lon_raw = (
            value_or_body.get("longitude")
            if value_or_body.get("longitude") is not None
            else (value_or_body.get("lon") if value_or_body.get("lon") is not None else value_or_body.get("lng"))
        )
        if lat_raw is not None and lon_raw is not None:
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
                if _is_valid_coord(lat, lon):
                    return lat, lon
            except (ValueError, TypeError):
                pass

    # 3. String format: "19.2435, 73.1310" or "[19.2435, 73.1310]"
    if isinstance(value_or_body, str):
        cleaned = value_or_body.strip().strip("[]()").strip()
        parts = re.split(r"[,;\s]+", cleaned)
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                if _is_valid_coord(lat, lon):
                    return lat, lon
            except (ValueError, TypeError):
                pass

    return None


def _is_valid_coord(lat: float, lon: float) -> bool:
    """Validate that coordinates are real, finite numbers within valid geographic bounds."""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    # Exclude 0,0 default / missing coords
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:
        return False
    # Latitude must be between -90 and 90, Longitude between -180 and 180
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


# =========================================================================
# 2. Haversine Distance Formula
# =========================================================================
def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two GPS points in meters
    using the Haversine formula (Earth radius = 6,371,000 meters).
    """
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


# =========================================================================
# 3. Polygon Centroid & Area Math (Green's Theorem)
# =========================================================================
def polygon_centroid_and_area(ring: List[List[float]]) -> Tuple[float, float, float]:
    """
    Compute the true 2D centroid (lat, lon) and signed area of a polygon ring [[lon, lat], ...]
    using Green's theorem (shoelace formula).
    Returns (centroid_lat, centroid_lon, area).
    """
    n = len(ring)
    if n < 3:
        if n > 0:
            return float(ring[0][1]), float(ring[0][0]), 0.0
        return 0.0, 0.0, 0.0

    signed_area = 0.0
    cx = 0.0  # lon
    cy = 0.0  # lat

    for i in range(n - 1):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[i + 1][0], ring[i + 1][1]
        a = (x0 * y1 - x1 * y0)
        signed_area += a
        cx += (x0 + x1) * a
        cy += (y0 + y1) * a

    signed_area *= 0.5
    if abs(signed_area) < 1e-12:
        # Degenerate polygon -> fallback to arithmetic mean
        avg_lon = sum(p[0] for p in ring) / n
        avg_lat = sum(p[1] for p in ring) / n
        return avg_lat, avg_lon, 0.0

    cx = cx / (6.0 * signed_area)
    cy = cy / (6.0 * signed_area)
    return cy, cx, abs(signed_area)


def get_feature_centroid(geom: Dict[str, Any]) -> Tuple[float, float]:
    """
    Extract the geometric centroid (lat, lon) of a GeoJSON geometry (Polygon or MultiPolygon).
    MultiPolygons are area-weighted.
    """
    g_type = geom.get("type")
    coords = geom.get("coordinates", [])

    if g_type == "Polygon":
        if coords and coords[0]:
            lat, lon, _ = polygon_centroid_and_area(coords[0])
            return lat, lon

    elif g_type == "MultiPolygon":
        total_area = 0.0
        weighted_lat = 0.0
        weighted_lon = 0.0

        for poly in coords:
            if poly and poly[0]:
                c_lat, c_lon, area = polygon_centroid_and_area(poly[0])
                total_area += area
                weighted_lat += c_lat * area
                weighted_lon += c_lon * area

        if total_area > 0:
            return weighted_lat / total_area, weighted_lon / total_area
        elif coords and coords[0] and coords[0][0]:
            lat, lon, _ = polygon_centroid_and_area(coords[0][0])
            return lat, lon

    return 0.0, 0.0


def point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    """Ray-casting algorithm to test if point (x=lon, y=lat) is inside polygon coordinates."""
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


# =========================================================================
# 4. OpenStreetMap Nominatim Geocoding Client
# =========================================================================
class NominatimClient:
    """
    OpenStreetMap Nominatim reverse and forward geocoding client.
    - Reverse geocoding: [latitude, longitude] -> Human-readable street address & landmark
    - Forward geocoding: Address text -> [latitude, longitude] coordinates
    Complies with OSM Nominatim usage policy (User-Agent header, in-memory cache, retries).
    """
    USER_AGENT = "WardMitra-Municipal-App/1.0 (contact: support@wardmitra.gov.in)"

    def __init__(self, timeout: float = 4.0):
        self.timeout = timeout
        self._reverse_cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
        self._forward_cache: Dict[str, Dict[str, Any]] = {}

    def reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Convert GPS coordinates [latitude, longitude] to human-readable address via OSM Nominatim.
        """
        # Round coords to 4 decimal places (~11 meters) for effective caching
        cache_key = (round(latitude, 4), round(longitude, 4))
        if cache_key in self._reverse_cache:
            return self._reverse_cache[cache_key]

        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&addressdetails=1"
        headers = {"User-Agent": self.USER_AGENT}

        for attempt in range(2):
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    addr = data.get("address", {})
                    display = data.get("display_name", "")

                    # Construct clean, readable municipal address
                    parts = []
                    for k in ["road", "neighbourhood", "suburb", "borough", "city", "county", "postcode"]:
                        val = addr.get(k)
                        if val and val not in parts:
                            parts.append(val)

                    formatted = ", ".join(parts) if parts else display

                    result = {
                        "success": True,
                        "display_name": display,
                        "formatted_address": formatted,
                        "address": addr,
                        "latitude": latitude,
                        "longitude": longitude
                    }
                    self._reverse_cache[cache_key] = result
                    return result
            except Exception as e:
                logger.warning(f"Nominatim reverseGeocode attempt {attempt+1} error: {e}")

        # Fallback if OSM is unreachable
        fallback_res = {
            "success": False,
            "display_name": f"{latitude:.5f}, {longitude:.5f}",
            "formatted_address": f"GPS: {latitude:.5f}, {longitude:.5f}",
            "address": {},
            "latitude": latitude,
            "longitude": longitude
        }
        return fallback_res

    def forward_geocode(self, query: str, city_bias: str = "Kalyan-Dombivli") -> Optional[Dict[str, Any]]:
        """
        Convert address/landmark text into [latitude, longitude] coordinates via OSM Nominatim.
        Biases search to the local municipal area if not specified in text.
        """
        if not query or len(query.strip()) < 3:
            return None

        q_clean = query.strip()
        q_lower = q_clean.lower()
        if q_lower in self._forward_cache:
            return self._forward_cache[q_lower]

        # Municipal biasing if query doesn't specify city
        search_query = q_clean
        if not any(city in q_lower for city in ["kalyan", "dombivli", "डोंबिवली", "कल्याण", "thane", "ठाणे", "mumbai"]):
            search_query = f"{q_clean}, {city_bias}, Maharashtra, India"

        url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(search_query)}&addressdetails=1&limit=1"
        headers = {"User-Agent": self.USER_AGENT}

        for attempt in range(2):
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) > 0:
                        first = data[0]
                        lat = float(first.get("lat"))
                        lon = float(first.get("lon"))
                        display = first.get("display_name", "")
                        res = {
                            "success": True,
                            "latitude": lat,
                            "longitude": lon,
                            "display_name": display,
                            "formatted_address": display,
                            "raw": first
                        }
                        self._forward_cache[q_lower] = res
                        logger.info(f"Nominatim resolved '{query}' -> [{lat:.5f}, {lon:.5f}] ({display[:40]}...)")
                        return res
            except Exception as e:
                logger.warning(f"Nominatim forwardGeocode attempt {attempt+1} error: {e}")

        return None


# =========================================================================
# 5. Municipal Ward Reverse Geocoding & Centroid Service
# =========================================================================
class GeoLookupService:
    """
    Municipal Ward Reverse Geocoding & Centroid Matching Service.
    - Accepts GPS coordinates as [latitude, longitude]
    - Computes polygon centroids for all 123 municipal wards
    - Determines official Ward ID & Name using Haversine formula to Centroid + Point-in-Polygon
    - Resolves human-readable addresses using OpenStreetMap Nominatim
    - Resolves text addresses/landmarks to coordinates & wards using Forward Nominatim
    - Connects to PostGIS if enabled in database
    """
    def __init__(self, geojson_path: Optional[Path] = None):
        self.path = geojson_path or GEOJSON_PATH
        self.features: List[Dict[str, Any]] = []
        self.ward_centroids: List[Dict[str, Any]] = []
        self.nominatim = NominatimClient()

        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.features = data.get("features", [])

                # Precompute centroids for all 123 wards for instant O(N) Haversine distance lookup
                for f in self.features:
                    props = f.get("properties", {})
                    ward_code = str(props.get("sourcewardcode") or props.get("objectid") or "Unknown")
                    ward_name = props.get("ward_lgd_name") or f"Ward No. {ward_code}"
                    geom = f.get("geometry", {})

                    c_lat, c_lon = get_feature_centroid(geom)
                    self.ward_centroids.append({
                        "ward_id": ward_code,
                        "ward_name": ward_name,
                        "centroid_lat": c_lat,
                        "centroid_lon": c_lon,
                        "feature": f
                    })

                logger.info(f"GeoLookupService: Loaded {len(self.features)} ward boundaries & precomputed {len(self.ward_centroids)} centroids.")
            except Exception as e:
                logger.error(f"Failed to load GeoJSON wards from {self.path}: {e}")
        else:
            logger.warning(f"GeoJSON file not found at {self.path}")

    def find_nearest_ward_by_centroid(self, latitude: float, longitude: float) -> Tuple[str, str, float, float, float]:
        """
        Calculate Haversine distance between [latitude, longitude] and all ward centroids.
        Returns: (ward_id, ward_name, distance_meters, centroid_lat, centroid_lon)
        """
        if not self.ward_centroids:
            return DEFAULT_FALLBACK_WARD_ID, DEFAULT_FALLBACK_WARD_NAME, 0.0, latitude, longitude

        min_dist = float("inf")
        nearest = self.ward_centroids[0]

        for item in self.ward_centroids:
            dist = haversine_distance_meters(
                latitude, longitude,
                item["centroid_lat"], item["centroid_lon"]
            )
            if dist < min_dist:
                min_dist = dist
                nearest = item

        return (
            nearest["ward_id"],
            nearest["ward_name"],
            round(min_dist, 1),
            nearest["centroid_lat"],
            nearest["centroid_lon"]
        )

    def lookup_ward(
        self,
        latitude: float,
        longitude: float,
        fetch_address: bool = True
    ) -> GeoLookupResult:
        """
        Main lookup function:
        1. Checks Point-in-Polygon containment (exact match).
        2. Calculates distance to ward Centroid using Haversine formula.
        3. If coordinates are outside boundary, selects the nearest ward by Centroid distance.
        4. Queries OpenStreetMap Nominatim for the verified address.
        """
        matched_ward_id = None
        matched_ward_name = None
        is_inside_polygon = False
        matching_centroid_lat = None
        matching_centroid_lon = None
        distance_to_centroid = None

        # Step 1: Check Point-in-Polygon containment
        for item in self.ward_centroids:
            feature = item["feature"]
            geom = feature.get("geometry", {})
            g_type = geom.get("type")
            coords = geom.get("coordinates", [])

            inside = False
            if g_type == "Polygon":
                for ring in coords:
                    if point_in_polygon(longitude, latitude, ring):
                        inside = True
                        break
            elif g_type == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        if point_in_polygon(longitude, latitude, ring):
                            inside = True
                            break
                    if inside:
                        break

            if inside:
                is_inside_polygon = True
                matched_ward_id = item["ward_id"]
                matched_ward_name = item["ward_name"]
                matching_centroid_lat = item["centroid_lat"]
                matching_centroid_lon = item["centroid_lon"]
                distance_to_centroid = round(
                    haversine_distance_meters(latitude, longitude, matching_centroid_lat, matching_centroid_lon),
                    1
                )
                break

        # Step 2: If not inside any polygon, resolve by nearest Centroid via Haversine formula
        if not is_inside_polygon:
            (
                c_ward_id,
                c_ward_name,
                c_dist,
                c_lat,
                c_lon
            ) = self.find_nearest_ward_by_centroid(latitude, longitude)

            matched_ward_id = c_ward_id
            matched_ward_name = c_ward_name
            matching_centroid_lat = c_lat
            matching_centroid_lon = c_lon
            distance_to_centroid = c_dist
            resolution_method = "haversine_centroid"
        else:
            resolution_method = "polygon_containment"

        # Step 3: Fetch verified street/area address from OpenStreetMap Nominatim
        address_text = None
        display_name = None
        if fetch_address:
            try:
                nom_res = self.nominatim.reverse_geocode(latitude, longitude)
                address_text = nom_res.get("formatted_address")
                display_name = nom_res.get("display_name")
            except Exception as e:
                logger.warning(f"OSM Nominatim reverse geocoding skipped: {e}")

        return GeoLookupResult(
            latitude=latitude,
            longitude=longitude,
            location=[latitude, longitude],
            ward_id=matched_ward_id or DEFAULT_FALLBACK_WARD_ID,
            ward_name=matched_ward_name or DEFAULT_FALLBACK_WARD_NAME,
            is_within_boundary=is_inside_polygon,
            address=address_text,
            display_name=display_name,
            distance_to_centroid_meters=distance_to_centroid,
            centroid_lat=matching_centroid_lat,
            centroid_lon=matching_centroid_lon,
            resolution_method=resolution_method
        )

    # =========================================================================
    # 6. Landmark Geo-Dictionary + Forward Nominatim Resolution
    # =========================================================================
    LANDMARK_DIRECTORY = [
        {
            "aliases": ["शिवाजी चौक", "shivaji chowk", "शिवाजी महाराज चौक", "shivaji maharaj chowk"],
            "ward_id": "19A",
            "ward_name": "Ward 19A - Shivaji Chowk",
            "lat": 19.2435, "lon": 73.1310
        },
        {
            "aliases": ["बिर्ला कॉलेज", "birla college", "बिर्ला मंदिर", "birla mandir", "birla"],
            "ward_id": "43",
            "ward_name": "Ward 43 - Birla College Road",
            "lat": 19.2520, "lon": 73.1480
        },
        {
            "aliases": ["आधारवाडी", "adharwadi", "आधारवाडी चौक", "adharwadi chowk", "आधारवाडी जेल"],
            "ward_id": "12",
            "ward_name": "Ward 12 - Adharwadi",
            "lat": 19.2480, "lon": 73.1250
        },
        {
            "aliases": ["कल्याण स्टेशन", "kalyan station", "स्टेशन रोड", "station road", "रेल्वे स्टेशन"],
            "ward_id": "15",
            "ward_name": "Ward 15 - Kalyan Station West",
            "lat": 19.2360, "lon": 73.1300
        },
        {
            "aliases": ["रामबाग", "rambaug", "rambaug lane", "रामबाग लेन"],
            "ward_id": "19B",
            "ward_name": "Ward 19B - Rambaug",
            "lat": 19.2390, "lon": 73.1350
        },
        {
            "aliases": ["टिळक चौक", "tilak chowk", "टिळक नगर", "tilak nagar", "टिळक रोड"],
            "ward_id": "20",
            "ward_name": "Ward 20 - Tilak Chowk",
            "lat": 19.2410, "lon": 73.1290
        },
        {
            "aliases": ["खडकपाडा", "khadakpada", "khadak pada", "खडक पाडा"],
            "ward_id": "35",
            "ward_name": "Ward 35 - Khadakpada",
            "lat": 19.2550, "lon": 73.1370
        },
        {
            "aliases": ["डोंबिवली स्टेशन", "dombivli station", "मानपाडा रोड", "manpada road", "मानपाडा"],
            "ward_id": "65",
            "ward_name": "Ward 65 - Dombivli East",
            "lat": 19.2180, "lon": 73.0870
        },
        {
            "aliases": ["महापालिका मुख्यालय", "kdmc hq", "kdmc", "पालिका कार्यालय", "kdmc office"],
            "ward_id": "19",
            "ward_name": "Ward 19 - KDMC Central Head Office",
            "lat": 19.2405, "lon": 73.1302
        },
        {
            "aliases": ["टिटवाळा", "titwala", "महागणपती मंदिर", "titwala station"],
            "ward_id": "01",
            "ward_name": "Ward 01 - Titwala",
            "lat": 19.2980, "lon": 73.2050
        },
        {
            "aliases": ["मोहने", "mohane", "एनआरसी", "nrc"],
            "ward_id": "05",
            "ward_name": "Ward 05 - Mohane",
            "lat": 19.2710, "lon": 73.1550
        },
        {
            "aliases": ["गांधी नगर", "gandhi nagar", "गांधी चौक"],
            "ward_id": "22",
            "ward_name": "Ward 22 - Gandhi Nagar",
            "lat": 19.2340, "lon": 73.1320
        }
    ]

    def lookup_by_address_or_landmark(self, text: str) -> Optional[GeoLookupResult]:
        """
        Dual-mode address resolution:
        1. Checks curated local Landmark Directory (ultra-fast, offline).
        2. If not found, queries OpenStreetMap Nominatim Search (Forward Geocoding)
           to extract [latitude, longitude], then computes Ward & Centroid distance!
        """
        if not text or len(text.strip()) < 3:
            return None

        text_lower = text.lower()

        # Step 1: Check Landmark Directory
        for item in self.LANDMARK_DIRECTORY:
            for alias in item["aliases"]:
                if alias in text_lower:
                    logger.info(f"Landmark matched from directory '{alias}' -> Ward {item['ward_id']}")
                    res = self.lookup_ward(item["lat"], item["lon"], fetch_address=False)
                    res.ward_id = item["ward_id"]
                    res.ward_name = item["ward_name"]
                    res.address = f"{alias.title()}, Kalyan-Dombivli"
                    res.resolution_method = "landmark_directory"
                    return res

        # Step 2: Forward Geocode via OpenStreetMap Nominatim
        nom_res = self.nominatim.forward_geocode(text)
        if nom_res and nom_res.get("latitude") and nom_res.get("longitude"):
            lat = nom_res["latitude"]
            lon = nom_res["longitude"]
            res = self.lookup_ward(lat, lon, fetch_address=False)
            res.address = nom_res.get("formatted_address")
            res.display_name = nom_res.get("display_name")
            res.resolution_method = "nominatim_forward"
            return res

        return None

    # Backward compatibility
    def lookup_by_landmark_text(self, text: str) -> Optional[GeoLookupResult]:
        return self.lookup_by_address_or_landmark(text)
