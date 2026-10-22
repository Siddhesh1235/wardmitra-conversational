"""
Builds municipal knowledge documents from live Node.js backend reference data,
in the same shape as MUNICIPAL_KNOWLEDGE_DOCS (id, keywords, title, content_mr/hi/en),
so the Hybrid RAG engine can answer ward / councillor questions from verified data.
"""
import re
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

TO_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def _seat_label(ward: Dict[str, Any]) -> str:
    """'Ward 12A' -> '12A'."""
    m = re.search(r"(\d+\s*[A-Za-z]?)\s*$", str(ward.get("name") or ""))
    return m.group(1).replace(" ", "") if m else str(ward.get("number"))


def _councillor(ward: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Prefer the councillor who registered on the app ('actual_*') over the reference listing."""
    return {
        "name": ward.get("actual_councillor_name") or ward.get("councillor_name"),
        "party": ward.get("councillor_party"),
        "phone": ward.get("actual_councillor_phone") or ward.get("councillor_phone") or ward.get("contact_number"),
    }


def build_ward_docs(wards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One document per prabhag (ward number), listing each seat (A-D) with its councillor,
    party and contact number. Wards with no councillor name are skipped.
    The city name is deliberately left out of the text: it would appear in every document and make
    any query that mentions the city rank these 31 documents above the relevant FAQ.
    """
    by_number: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for w in wards:
        if w.get("number") is not None and (w.get("councillor_name") or w.get("actual_councillor_name")):
            by_number[int(w["number"])].append(w)

    docs = []
    for number in sorted(by_number):
        seats = sorted(by_number[number], key=_seat_label)
        n_dev = str(number).translate(TO_DEVANAGARI_DIGITS)

        lines_en, lines_mr, lines_hi, names = [], [], [], []
        for w in seats:
            c = _councillor(w)
            seat = _seat_label(w)
            party = f" ({c['party']})" if c["party"] else ""
            phone_en = f" – Phone: {c['phone']}" if c["phone"] else ""
            phone_mr = f" – संपर्क: {c['phone']}" if c["phone"] else ""
            phone_hi = f" – संपर्क: {c['phone']}" if c["phone"] else ""
            lines_en.append(f"- Ward {seat}: {c['name']}{party}{phone_en}")
            lines_mr.append(f"- वॉर्ड {seat}: {c['name']}{party}{phone_mr}")
            lines_hi.append(f"- वार्ड {seat}: {c['name']}{party}{phone_hi}")
            names.append(str(c["name"]))

        seat_range = f"{_seat_label(seats[0])}–{_seat_label(seats[-1])}" if len(seats) > 1 else _seat_label(seats[0])
        docs.append({
            "id": f"backend_prabhag_{number}",
            "source": "backend:/wards",
            "keywords": [
                f"ward {number}", f"prabhag {number}", f"प्रभाग {number}", f"प्रभाग {n_dev}",
                f"वॉर्ड {number}", f"वॉर्ड {n_dev}", f"वार्ड {number}", f"वार्ड {n_dev}",
                "councillor", "corporator", "nagarsevak", "नगरसेवक", "नगरसेविका", "पार्षद",
                *[_seat_label(w) for w in seats], *names,
            ],
            "title": f"प्रभाग {number} ({seat_range}) नगरसेवक व संपर्क (Ward {number} Councillors & Contacts)",
            "content_mr": (
                f"प्रभाग क्र. {number} मध्ये {len(seats)} नगरसेवक (वॉर्ड {seat_range}) आहेत:\n"
                + "\n".join(lines_mr)
            ),
            "content_hi": (
                f"प्रभाग क्र. {number} में {len(seats)} पार्षद (वार्ड {seat_range}) हैं:\n"
                + "\n".join(lines_hi)
            ),
            "content_en": (
                f"Prabhag (Ward) {number} has {len(seats)} councillors (wards {seat_range}):\n"
                + "\n".join(lines_en)
            ),
        })

    logger.info(f"kb_sync: built {len(docs)} prabhag documents from {len(wards)} backend ward records.")
    return docs


def fetch_backend_knowledge_docs(backend_client: Any) -> List[Dict[str, Any]]:
    """Fetch reference data from the backend and convert it to knowledge documents. Never raises."""
    try:
        return build_ward_docs(backend_client.list_wards())
    except Exception as e:
        logger.warning(f"kb_sync: could not build backend knowledge docs ({e}).")
        return []
