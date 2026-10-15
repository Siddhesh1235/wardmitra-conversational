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
    },
    {
        "id": "potholes_road_repair",
        "keywords": ["pothole", "road", "khadde", "खड्डा", "खड्डे", "रस्ता", "डांबर", "asphalt", "d दुरुस्ती"],
        "title": "Road Repair & Potholes Filing Guidelines",
        "content_mr": "रस्त्यावरील खड्डे आणि रस्ते दुरुस्तीचे काम शहर अभियंता व सार्वजनिक बांधकाम विभागामार्फत केले जाते. पावसाळ्यादरम्यान कोल्ड मिक्स / हॉट मिक्स पद्धतीने ४८ ते ७२ तासांत खड्डे बुजवले जातात. मोठा रस्ता खराब असल्यास संबंधित प्रभाग कार्यालयात संपर्क साधावा.",
        "content_en": "Road repairs and pothole filling are executed by the Municipal Civil Works Department. During monsoon, potholes are filled using cold mix/hot mix within 48 to 72 hours SLA."
    },
    {
        "id": "drainage_waterlogging",
        "keywords": ["drainage", "gutter", "gatar", "गटार", "ड्रेनेज", "पाणी तुंबणे", "waterlogging", "चोकप", "नाला"],
        "title": "Drainage Cleaning & Waterlogging Resolution",
        "content_mr": "नाले व गटारे सफाई मोहीम पावसाळ्यापूर्वी आणि नियमितपणे राबवली जाते. गटार तुंबल्यास किंवा घाण पाणी रस्त्यावर आल्यास २४ ते ४८ तासांत सफाई पथक पाठवले जाते. तातडीच्या मदतीसाठी आपत्ती व्यवस्थापन कक्षाशी (०२५१-२२०४०६०) संपर्क साधावा.",
        "content_en": "Drainage and gutter desilting operations run regularly. Choked drainage or sewage overflows are cleared within 24 to 48 hours. For emergency flood/waterlogging, contact the Disaster Management Cell."
    },
    {
        "id": "water_supply_timings",
        "keywords": ["water", "pani", "पाणी", "पाणीपुरवठा", "water timing", "गळती", "कमी दाब", "tanker", "टँकर"],
        "title": "Municipal Water Supply Timings & Emergency Tanker",
        "content_mr": "महापालिकेमार्फत दिवसातून एकदा निर्धारित वेळेनुसार पाणीपुरवठा केला जातो. पाईपलाईन गळती १२ ते २४ तासांत दुरुस्त केली जाते. पाणीपुरवठा खंडित झाल्यास किंवा टँकरची गरज असल्यास ०२५१-२२११८८८ वर संपर्क साधता येतो.",
        "content_en": "Water supply is provided once daily per ward zone schedules. Main pipe bursts and leakages are resolved within 12 to 24 hours. Emergency water tanker requests can be made via 0251-2211888."
    },
    {
        "id": "dengue_malaria_fogging",
        "keywords": ["fogging", "mosquito", "डास", "फवारणी", "dengue", "malaria", "डेंग्यू", "मलेरिया", "रोगराई"],
        "title": "Vector Control, Sanitation & Fogging",
        "content_mr": "डासांचा प्रादुर्भाव रोखण्यासाठी आरोग्य विभागामार्फत धूर फवारणी (Fogging) आणि जंतूनाशक पावडर टाकली जाते. आपल्या परिसरात फवारणी हवी असल्यास वॉर्ड आरोग्य निरीक्षकाकडे किंवा १८००-२३३-००५० वर नोंद करावी.",
        "content_en": "Health department conducts chemical fogging and anti-larval spraying to prevent vector-borne diseases (Dengue, Malaria). Citizens can request fogging via the Ward Health Inspector."
    },
    {
        "id": "rti_application",
        "keywords": ["rti", "माहिती अधिकार", "right to information", "अर्ज", "माहिती अधिकारी"],
        "title": "Right to Information (RTI) Procedures",
        "content_mr": "माहिती अधिकार कायद्यांतर्गत (RTI) अर्ज करण्यासाठी विहित नमुन्यातील अर्ज १० रुपयांच्या कोर्ट फी स्टॅम्पसह जनमाहिती अधिकाऱ्याकडे (PIO) सादर करावा लागतो किंवा 'RTI Online Maharashtra' पोर्टलवर ऑनलाइन अर्ज करता येतो.",
        "content_en": "Applications under the Right to Information Act (RTI) can be submitted to the Public Information Officer (PIO) with a Rs. 10 court fee stamp or online through the RTI Online Maharashtra portal."
    }
]

class MunicipalKnowledgeBase:
    """
    Search and retrieve verified municipal knowledge, FAQs, and contact directories.
    Powered by Hybrid RAG (BM25 lexical + OpenAI vector embeddings + RRF ranking)
    with resilient fallback to fast in-memory keyword matching.
    """
    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None, openai_api_key: Optional[str] = None):
        self.docs = docs or MUNICIPAL_KNOWLEDGE_DOCS
        self.hybrid_engine = None
        try:
            from app.tools.hybrid_rag import HybridMunicipalKnowledgeBase
            self.hybrid_engine = HybridMunicipalKnowledgeBase(docs=self.docs, openai_api_key=openai_api_key)
            logger.info("MunicipalKnowledgeBase: Successfully initialized Hybrid RAG engine.")
        except Exception as e:
            logger.warning(f"MunicipalKnowledgeBase: Fallback to keyword mode ({e}).")

    def search(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Search knowledge base using Hybrid RAG (BM25 + pgvector / vector embeddings).
        Falls back to rule-based keyword matching if hybrid engine is unavailable.
        """
        if not query:
            return {"found": False, "results": []}

        # Try Hybrid Engine first
        if self.hybrid_engine:
            try:
                res = self.hybrid_engine.search(query, top_k=top_k)
                if res.get("found"):
                    return res
            except Exception as e:
                logger.warning(f"Hybrid RAG search error: {e}. Using keyword fallback.")

        # Resilient local keyword fallback
        q_lower = query.lower()
        scored_docs = []

        for doc in self.docs:
            score = 0
            for kw in doc.get("keywords", []):
                kw_lower = kw.lower()
                if kw_lower in q_lower:
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

