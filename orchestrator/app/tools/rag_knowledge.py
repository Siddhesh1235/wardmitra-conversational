"""
Municipal Knowledge Base (RAG & Civic FAQ tool).
Provides curated official municipal policies, ward office timings, 
emergency contacts, standard grievance resolution SLAs, and citizen service details.
"""
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MUNICIPAL_KNOWLEDGE_DOCS = [
    {
        "id": "office_timings",
        "keywords": ["office", "timing", "time", "open", "close", "वेळ", "कार्यालय", "कधी उघडते", "बंद", "सुट्टी"],
        "title": "KDMC Office Timings & Working Days",
        "content_mr": "महापालिका आणि वॉर्ड कार्यालये सोमवार ते शनिवार सकाळी १०:०० ते सायंकाळी ५:४५ पर्यंत उघडी असतात. नागरिकांसाठी भेटीची वेळ दुपारी ३:०० ते ५:०० आहे. दुसरा आणि चौथा शनिवार तसेच रविवार व शासकीय सुट्ट्यांच्या दिवशी कार्यालय बंद असते.",
        "content_en": "Municipal and ward offices operate Monday through Saturday from 10:00 AM to 5:45 PM. Public visiting hours are 3:00 PM to 5:00 PM. Closed on 2nd and 4th Saturdays, Sundays, and public holidays."
    },
    {
        "id": "ward_19a_office",
        "keywords": ["ward 19a", "ward office", "कल्याण", "डोंबिवली", "कार्यालय पत्ता", "नगरसेवक", "councillor", "पत्ता", "address"],
        "title": "Ward 19A Office & Administration",
        "content_mr": "वॉर्ड १९ए (कल्याण-डोंबिवली) कार्यालय: वॉर्ड कार्यालय, प्रभाग समिती कार्यालय, नागरिक सुविधा केंद्र जवळ. येथे जन्म-मृत्यू दाखले, कर संकलन आणि स्थानिक नागरी तक्रारी स्वीकारल्या जातात.",
        "content_en": "Ward 19A Office: Ward Citizen Facilitation Centre, Kalyan-Dombivli. Handles birth/death certificates, property tax collection, and civic grievance resolution."
    },
    {
        "id": "emergency_helpline",
        "keywords": ["emergency", "helpline", "phone", "number", "contact", "आपत्कालीन", "फोन", "नंबर", "संपर्क", "फायर", "पोलीस", "अग्निशामक"],
        "title": "Emergency & Civic Helplines",
        "content_mr": "आपत्कालीन संपर्क क्रमांक: \n- आपत्ती व्यवस्थापन कक्ष: ०२५१-२२०४०६० / २२०६२०६\n- अग्निशामक दल: १०१\n- पोलीस: १०० / ११२\n- रुग्णवाहिका: १०८\n- महावितरण (वीज तक्रार): १९१२ / १८००-२१२-३४३५\n- पाणीपुरवठा तक्रार: ०२५१-२२११८८८",
        "content_en": "Emergency Helplines: \n- Disaster Management Cell: 0251-2204060 / 2206206\n- Fire: 101\n- Police: 100 / 112\n- Ambulance: 108\n- Electricity (MSEDCL): 1912 / 1800-212-3435\n- Water Supply Emergency: 0251-2211888"
    },
    {
        "id": "grievance_sla",
        "keywords": ["sla", "time", "days", "kiti divas", "किती दिवस", "वेळ लागेल", "कधी पूर्ण होणार", "निकालाची वेळ", "कचरा कधी उचलणार"],
        "title": "Standard Grievance Resolution Timelines (SLA)",
        "content_mr": "तक्रार निवारण प्रमाणभूत कालावधी (SLA):\n- कचरा उचलणे / स्वच्छता: २४ तास\n- पथदिवे (स्ट्रीटलाईट): २४ ते ४८ तास\n- ड्रेनेज / सांडपाणी तुंबणे: २४ ते ४८ तास\n- रस्त्यावरील खड्डे: ४८ ते ७२ तास\n- पाणी गळती दुरुस्ती: १२ ते २४ तास",
        "content_en": "Citizen Charter Grievance Resolution SLAs:\n- Garbage Collection / Sanitation: 24 hours\n- Streetlights / Electrical poles: 24 to 48 hours\n- Drainage / Sewage overflows: 24 to 48 hours\n- Road Potholes: 48 to 72 hours\n- Water pipe leakages: 12 to 24 hours"
    },
    {
        "id": "property_tax_water_bill",
        "keywords": ["tax", "property tax", "water bill", "पाणीपट्टी", "घरपट्टी", "कर", "बिल", "payment", "भरणा"],
        "title": "Property Tax & Water Bill Payments",
        "content_mr": "घरपट्टी (मालमत्ता कर) आणि पाणीपट्टी भरण्यासाठी महापालिकेच्या अधिकृत पोर्टलवर (kdmc.gov.in) किंवा जवळच्या नागरिक सुविधा केंद्रात (CFC) ऑनलाइन/ऑफलाइन भरणा करता येतो. वेळेत भरणा केल्यास सवलत मिळते.",
        "content_en": "Property tax and water bills can be paid online at the official municipal portal (kdmc.gov.in) or offline at Citizen Facilitation Centres (CFC). Rebates apply for early payments."
    },
    {
        "id": "birth_death_certificate",
        "keywords": ["birth", "death", "certificate", "दाखला", "जन्म", "मृत्यू", "प्रमाणपत्र"],
        "title": "Birth and Death Certificate Application",
        "content_mr": "जन्म आणि मृत्यू नोंदणी प्रमाणपत्र मिळवण्यासाठी हॉस्पिटलचा डिस्चार्ज/नोंदणी अहवाल घेऊन संबंधित वॉर्ड कार्यालयातील आरोग्य विभागात किंवा 'आपले सरकार' पोर्टलवर ऑनलाइन अर्ज करता येतो. २१ दिवसांच्या आत नोंदणी करणे मोफत असते.",
        "content_en": "Birth and death certificates can be obtained from the Ward Health Department or online via Aaple Sarkar portal with valid hospital discharge/reporting records. Free within 21 days of occurrence."
    }
]

class MunicipalKnowledgeBase:
    """
    Search and retrieve verified municipal knowledge, FAQs, and contact directories.
    """
    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None):
        self.docs = docs or MUNICIPAL_KNOWLEDGE_DOCS

    def search(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Search knowledge base using keyword matching and relevance scoring.
        Returns top matched snippets in Marathi and English.
        """
        if not query:
            return {"found": False, "results": []}

        q_lower = query.lower()
        scored_docs = []

        for doc in self.docs:
            score = 0
            for kw in doc["keywords"]:
                kw_lower = kw.lower()
                if kw_lower in q_lower:
                    # Exact word boundary match gets higher weight
                    if re.search(rf"\b{re.escape(kw_lower)}\b", q_lower):
                        score += 3
                    else:
                        score += 1

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_results = [item[1] for item in scored_docs[:top_k]]

        if not top_results:
            return {
                "found": False,
                "message": "या विषयाबद्दल थेट माहिती उपलब्ध नाही, तरी वॉर्ड कार्यालयात किंवा १८००-२३३-००५० वर संपर्क साधावा.",
                "results": []
            }

        return {
            "found": True,
            "results": top_results
        }
