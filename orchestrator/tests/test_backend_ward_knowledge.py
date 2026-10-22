from unittest.mock import MagicMock

from app.tools.hybrid_rag import BM25Scorer
from app.tools.kb_sync import build_ward_docs, fetch_backend_knowledge_docs
from app.tools.rag_knowledge import MUNICIPAL_KNOWLEDGE_DOCS

# Shape mirrors the public GET /wards response
WARDS = [
    {"id": 58, "name": "Ward 1A", "number": 1, "councillor_name": "Varun Patil", "councillor_party": "BJP",
     "councillor_phone": "9763333376", "contact_number": "9763333376", "city_name": "Kalyan-Dombivli",
     "actual_councillor_name": None, "actual_councillor_phone": None},
    {"id": 59, "name": "Ward 1B", "number": 1, "councillor_name": "Sunita Kale", "councillor_party": "Shiv Sena",
     "councillor_phone": None, "contact_number": None, "city_name": "Kalyan-Dombivli",
     "actual_councillor_name": "Sunita R. Kale", "actual_councillor_phone": "9000000001"},
    {"id": 99, "name": "Ward 12C", "number": 12, "councillor_name": "Amit Joshi", "councillor_party": "INC",
     "councillor_phone": "9000000012", "contact_number": None, "city_name": "Kalyan-Dombivli",
     "actual_councillor_name": None, "actual_councillor_phone": None},
    {"id": 100, "name": "Ward 13A", "number": 13, "councillor_name": None, "councillor_party": None,
     "councillor_phone": None, "contact_number": None, "city_name": "Kalyan-Dombivli",
     "actual_councillor_name": None, "actual_councillor_phone": None},
]


def test_one_doc_per_prabhag_and_skips_empty():
    docs = build_ward_docs(WARDS)
    assert [d["id"] for d in docs] == ["backend_prabhag_1", "backend_prabhag_12"]


def test_doc_content_prefers_registered_councillor():
    doc = build_ward_docs(WARDS)[0]
    assert "Ward 1A: Varun Patil (BJP) – Phone: 9763333376" in doc["content_en"]
    assert "Ward 1B: Sunita R. Kale (Shiv Sena) – Phone: 9000000001" in doc["content_en"]
    assert "वॉर्ड 1A: Varun Patil" in doc["content_mr"]
    for key in ("title", "content_mr", "content_hi", "content_en", "keywords"):
        assert doc[key]


def test_bm25_finds_ward_councillor_doc():
    docs = MUNICIPAL_KNOWLEDGE_DOCS + build_ward_docs(WARDS)
    bm25 = BM25Scorer(docs)
    for query in ["who is the councillor of ward 12", "प्रभाग १२ नगरसेवक कोण आहेत", "Amit Joshi phone number"]:
        top_idx = bm25.score(query)[0][0]
        assert docs[top_idx]["id"] == "backend_prabhag_12", query


def test_fetch_backend_docs_never_raises():
    client = MagicMock()
    client.list_wards.side_effect = RuntimeError("backend down")
    assert fetch_backend_knowledge_docs(client) == []
