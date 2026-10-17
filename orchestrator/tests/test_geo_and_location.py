import pytest
from app.pipeline.geo_lookup import (
    parse_numeric_location,
    haversine_distance_meters,
    polygon_centroid_and_area,
    GeoLookupService,
    NominatimClient
)
from app.slot_checker.deterministic_gate import SlotChecker

def test_parse_numeric_location():
    # 1. Standard list [lat, lon]
    res1 = parse_numeric_location([19.2435, 73.1310])
    assert res1 == (19.2435, 73.1310)

    # 2. Tuple (lat, lon)
    res2 = parse_numeric_location((19.2500, 73.1400))
    assert res2 == (19.2500, 73.1400)

    # 3. Dict with lat/lng
    res3 = parse_numeric_location({"lat": 19.23, "lng": 73.12})
    assert res3 == (19.23, 73.12)

    # 4. Dict with latitude/longitude
    res4 = parse_numeric_location({"latitude": 19.234, "longitude": 73.125})
    assert res4 == (19.234, 73.125)

    # 5. Nested location dict
    res5 = parse_numeric_location({"location": [19.22, 73.08]})
    assert res5 == (19.22, 73.08)

    # 6. String formats
    res6 = parse_numeric_location("19.2435, 73.1310")
    assert res6 == (19.2435, 73.1310)

    res7 = parse_numeric_location("[19.2435, 73.1310]")
    assert res7 == (19.2435, 73.1310)

    # 7. Invalid coordinates should return None
    assert parse_numeric_location(None) is None
    assert parse_numeric_location([0.0, 0.0]) is None
    assert parse_numeric_location("invalid") is None
    assert parse_numeric_location([999.0, 73.0]) is None  # invalid latitude

def test_haversine_distance():
    # Distance between same points is 0
    assert haversine_distance_meters(19.2435, 73.1310, 19.2435, 73.1310) == 0.0
    # Approx 1 km difference in latitude (~1/111 degrees)
    dist = haversine_distance_meters(19.2400, 73.1300, 19.2490, 73.1300)
    assert 990 <= dist <= 1015

def test_polygon_centroid():
    # Square polygon from (0,0) to (2,2) with lon=x, lat=y
    ring = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
    lat, lon, area = polygon_centroid_and_area(ring)
    assert abs(lat - 1.0) < 1e-4
    assert abs(lon - 1.0) < 1e-4
    assert abs(area - 4.0) < 1e-4

def test_geo_lookup_centroids_loaded():
    geo_service = GeoLookupService()
    assert len(geo_service.features) > 100
    assert len(geo_service.ward_centroids) == len(geo_service.features)

    # Each centroid must have valid coordinates
    for item in geo_service.ward_centroids[:10]:
        assert 19.1 <= item["centroid_lat"] <= 19.4
        assert 73.0 <= item["centroid_lon"] <= 73.3
        assert item["ward_id"] is not None

def test_find_nearest_ward_by_centroid():
    geo_service = GeoLookupService()
    # Coordinates for Kalyan Shivaji Chowk area
    ward_id, ward_name, dist, c_lat, c_lon = geo_service.find_nearest_ward_by_centroid(19.2435, 73.1310)
    assert ward_id is not None
    assert dist > 0
    assert dist < 1000  # Within 1 km of central ward centroid

def test_lookup_ward_polygon_and_centroid():
    geo_service = GeoLookupService()

    # Point inside Kalyan Dombivli boundary
    res_inside = geo_service.lookup_ward(19.2435, 73.1310, fetch_address=False)
    assert res_inside.ward_id is not None
    assert res_inside.distance_to_centroid_meters is not None
    assert res_inside.centroid_lat is not None
    assert res_inside.centroid_lon is not None
    assert res_inside.location == [19.2435, 73.1310]

    # Point outside boundary -> Resolved via Haversine Centroid distance
    res_outside = geo_service.lookup_ward(19.3800, 73.2800, fetch_address=False)
    assert res_outside.is_within_boundary is False
    assert res_outside.resolution_method == "haversine_centroid"
    assert res_outside.ward_id is not None
    assert res_outside.distance_to_centroid_meters > 2000

def test_landmark_directory_and_forward_lookup():
    geo_service = GeoLookupService()
    res = geo_service.lookup_by_address_or_landmark("शिवाजी चौक")
    assert res is not None
    assert res.ward_id == "19A"
    assert res.resolution_method == "landmark_directory"

def test_slot_checker_with_numeric_location():
    slot_checker = SlotChecker()
    slots = {}

    # Pass mobile app location as [lat, lon]
    updated = slot_checker.extract_and_merge_slots(
        existing_slots=slots,
        incoming_location=[19.2435, 73.1310]
    )

    assert updated.get("has_gps") is True
    assert updated.get("latitude") == 19.2435
    assert updated.get("longitude") == 73.1310
    assert updated.get("location_coords") == [19.2435, 73.1310]
    assert updated.get("ward_id") is not None
    assert updated.get("distance_to_centroid_meters") is not None
    assert "location" in updated
