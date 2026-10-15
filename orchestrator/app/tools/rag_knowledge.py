"""
Municipal Knowledge Base (RAG & Civic FAQ tool).
Provides curated official municipal policies, filing channels, mandatory criteria,
escalation ladders, department classifications, emergency helplines, and glossary
in Marathi (देवनागरी), Hindi (हिंदी), and English.
"""
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MUNICIPAL_KNOWLEDGE_DOCS = [
    # =========================================================================
    # 1. Citizen Grievance Filing & Process FAQ (नागरी तक्रार नोंदणी मार्गदर्शक)
    # =========================================================================
    {
        "id": "how_to_file_complaint",
        "keywords": [
            "file complaint", "register complaint", "तक्रार नोंदणी", "कशी नोंदवायची", "शिकायत कैसे करें",
            "whatsapp bot", "web portal", "mobile app", "how to lodge", "madhyam", "माध्यम", "portal"
        ],
        "title": "वॉर्डमित्र द्वारे नागरिक तक्रार नोंदणीची माध्यमे (Filing Channels)",
        "content_mr": "वॉर्डमित्र द्वारे नागरिक २ प्रमुख माध्यमांतून तक्रार नोंदवू शकतात:\n१. व्हॉट्सॲप तक्रार निवारण बॉट (WhatsApp Bot): अधिकृत वॉर्डमित्र व्हॉट्सॲप क्रमांकावर समस्या, फोटो, लँडमार्क किंवा व्हॉईस नोट पाठवून. एआय बॉट मराठी, हिंदी किंवा इंग्रजीतून माहिती गोळा करून तक्रार नोंदवतो.\n२. नागरिक वेब पोर्टल / मोबाईल ॲप (Web Portal / Mobile App): संभाषणात्मक एआय चॅट किंवा थेट तक्रार फॉर्म भरून.",
        "content_hi": "वार्डमित्र के माध्यम से नागरिक २ प्रमुख माध्यमों से शिकायत दर्ज कर सकते हैं:\n१. व्हाट्सएप बॉट (WhatsApp Bot): आधिकारिक वार्डमित्र व्हाट्सएप नंबर पर समस्या, फोटो, लैंडमार्क या वॉयस नोट भेजकर। एआई बॉट हिंदी, मराठी या अंग्रेजी में जानकारी एकत्र कर शिकायत दर्ज करता है।\n२. नागरिक वेब पोर्टल / मोबाइल ऐप (Web Portal / Mobile App): संवादात्मक एआई चैट या सीधे शिकायत फॉर्म भरकर।",
        "content_en": "Citizens can lodge grievances through 2 primary channels:\n1. WhatsApp Bot: By sending complaint description, photo, landmark, or voice note to the official WardMitra WhatsApp number. The AI bot collects missing slots in Marathi, Hindi, or English.\n2. Citizen Web Portal / Mobile App: Via conversational AI chat or direct grievance forms."
    },
    {
        "id": "mandatory_complaint_info",
        "keywords": [
            "mandatory", "required info", "आवश्यक माहिती", "अनिवार्य जानकारी", "कोणती माहिती पाहिजे",
            "lagel", "document", "requirements", "details needed", "चार गोष्टी"
        ],
        "title": "तक्रार नोंदणीसाठी आवश्यक ४ गोष्टी (Mandatory Requirements)",
        "content_mr": "तक्रार दाखल करण्यासाठी खालील ४ गोष्टी आवश्यक (Mandatory) आहेत:\n१. समस्येचे स्पष्ट वर्णन (Clear Description): उदा. रस्त्यावर खड्डा, कचऱ्याचा ढीग, ड्रेनेज तुंबणे, पथदिवा बंद.\n२. अचूक पत्ता किंवा लँडमार्क (Accurate Landmark / GPS Location Pin): रस्त्याचे नाव, जवळचे दुकान, मंदिर, किंवा GPS लोकेशन पिन.\n३. समस्येचा प्रकार / प्रवर्ग (Category): उदा. रस्ते, घनकचरा, पाणीपुरवठा, ड्रेनेज, वीज.\n४. प्रत्यक्ष फोटो (Photographic Proof): समस्येचा आणि आजूबाजूच्या परिसराचा स्पष्ट फोटो.",
        "content_hi": "शिकायत दर्ज करने के लिए निम्नलिखित ४ विवरण अनिवार्य (Mandatory) हैं:\n१. समस्या का स्पष्ट विवरण: उदा. सड़क पर गड्ढा, कचरे का ढेर, नाली चोक, स्ट्रीट लाइट बंद।\n२. सटीक लैंडमार्क या GPS लोकेशन: सड़क का नाम, नजदीकी दुकान, मंदिर या GPS लोकेशन पिन।\n३. समस्या की श्रेणी (Category): जैसे सड़क, ठोस कचरा, जलापूर्ति, सीवेज, बिजली।\n४. प्रत्यक्ष फोटो (Photographic Proof): समस्या और आसपास के परिसर का स्पष्ट फोटो।",
        "content_en": "Four mandatory details required to lodge a civic grievance:\n1. Clear Description: Pothole, garbage dump, drainage overflow, dark streetlight.\n2. Accurate Landmark or GPS Location Pin: Street name, nearby shop, temple, or live pin.\n3. Problem Category: Roads, solid waste, water supply, drainage, electricity.\n4. Photographic Proof: Clear on-site photo showing the issue and surroundings."
    },
    {
        "id": "why_photo_mandatory",
        "keywords": [
            "photo mandatory", "photo ka pahije", "फोटो का आवश्यक", "फोटो क्यों जरूरी है", "proof", "image required", "bandhankarak"
        ],
        "title": "तक्रारीसोबत फोटो पाठवणे का बंधनकारक आहे? (Why Photo is Required)",
        "content_mr": "तक्रारीसोबत फोटो पाठवणे खालील कारणांसाठी बंधनकारक आहे:\n- समस्येचे गांभीर्य आणि व्याप्ती (Severity) त्वरित तपासण्यासाठी.\n- तक्रार खोटी किंवा स्पॅम नसल्याची खात्री (Verification) करण्यासाठी.\n- कनिष्ठ अभियंत्यांना (Junior Engineer) जागेवर जाण्यापूर्वी योग्य साहित्य व कर्मचारी पाठवणे सोपे व्हावे यासाठी.",
        "content_hi": "शिकायत के साथ फोटो भेजना निम्नलिखित कारणों से अनिवार्य है:\n- समस्या की गंभीरता और दायरा (Severity) तुरंत जांचने के लिए।\n- शिकायत फर्जी या स्पैम न होने की पुष्टि (Verification) करने के लिए।\n- कनिष्ठ अभियंता (JE) को मौके पर जाने से पहले सही उपकरण और कर्मचारी भेजना आसान बनाने के लिए।",
        "content_en": "Attaching a photo is mandatory for:\n- Instant assessment of problem severity and scope.\n- Verifying authenticity and eliminating spam or fake reports.\n- Assisting Junior Engineers (JE) in dispatching the exact tools, machinery, and crew."
    },
    {
        "id": "complaint_updates_process",
        "keywords": [
            "updates", "ticket id", "तिकीट क्रमांक", "अपडेट्स कसे मिळतात", "अपडेट कैसे मिलेगा", "sms", "whatsapp message", "stages", "wm-2026"
        ],
        "title": "तक्रारीचे अपडेट्स आणि टप्पे (Complaint Tracking Updates & Stages)",
        "content_mr": "तक्रार नोंदवल्यावर SMS आणि WhatsApp द्वारे अधिकृत तिकीट क्रमांक (उदा. WM-2026-XXXX) मिळतो. खालील ४ टप्प्यांवर अपडेट्स मिळतात:\n१. कनिष्ठ अभियंत्याकडे (JE) तक्रार सोपवली गेल्यावर (Assigned).\n२. प्रत्यक्ष जागेवर काम सुरू झाल्यावर (In Progress).\n३. काम पूर्ण झाल्यावर (कामाचा प्रत्यक्ष फोटो जोडला जातो).\n४. तक्रार अधिकृतपणे बंद (Closed) झाल्यावर.",
        "content_hi": "शिकायत दर्ज होते ही SMS और WhatsApp द्वारा आधिकारिक टिकट संख्या (उदा. WM-2026-XXXX) प्राप्त होती है। ४ चरणों में अपडेट्स मिलते हैं:\n१. कनिष्ठ अभियंता (JE) को कार्य सौंपे जाने पर (Assigned)।\n२. मौके पर वास्तविक कार्य प्रारंभ होने पर (In Progress)।\n३. कार्य पूर्ण होने पर (समाधान का वास्तविक फोटो संलग्न किया जाता है)।\n४. शिकायत आधिकारिक तौर पर बंद (Closed) होने पर।",
        "content_en": "Upon registration, citizens receive an official ticket ID (e.g. WM-2026-XXXX) via SMS and WhatsApp. Progress updates are sent at 4 stages:\n1. Assigned to Junior Engineer (JE) / Sanitary Inspector.\n2. Work started on-site (In Progress).\n3. Work completed (with on-site resolution photo proof).\n4. Officially closed."
    },
    {
        "id": "track_complaint_status",
        "keywords": [
            "status", "track", "स्थिती", "तक्रार स्थिती", "शिकायत की स्थिति", "live status", "check status", "ticket status", "माझी तक्रार", "meri shikayat"
        ],
        "title": "तक्रारीची सद्यस्थिती (Live Status) तपासणे",
        "content_mr": "नागरिक चॅटमध्ये किंवा पोर्टलवर आपला 'तक्रार तिकीट क्रमांक' (Ticket ID) टाकून किंवा 'माझ्या तक्रारीची स्थिती काय आहे?' असे विचारून थेट महापालिका डेटाबेसमधून लाइव्ह स्टेटस तपासू शकतात.",
        "content_hi": "नागरिक चैट में अपना 'शिकायत टिकट नंबर' डालकर या 'मेरी शिकायत की स्थिति क्या है?' पूछकर सीधे नगरपालिका डेटाबेस से लाइव स्थिति की जांच कर सकते हैं।",
        "content_en": "Citizens can check real-time grievance status by entering their Ticket ID in chat/portal or asking 'What is the status of my complaint?' to retrieve live data from the municipal database."
    },
    {
        "id": "duplicate_detection_clustering",
        "keywords": [
            "duplicate", "cluster", "एकाच समस्येबाबत", "डुप्लिकेट तक्रार", "एक ही समस्या", "duplicate detection", "500 meter", "ekach khadda", "do baar"
        ],
        "title": "एकाच समस्येबाबत अनेक नागरिकांनी तक्रार केल्यास (Duplicate Clustering)",
        "content_mr": "५०० मीटरच्या परिघात एकाच खड्ड्याची किंवा कचऱ्याची तक्रार आल्यास सिस्टीम त्या तक्रारी एकत्र (Cluster) करते. सर्व तक्रारदारांना स्वतंत्र अपडेट्स मिळतात, मात्र ठेकेदाराला एकच वर्क ऑर्डर दिली जाते जेणेकरून कामाचा आणि बिलाचा दुहेरीपणा टाळता येतो.",
        "content_hi": "५०० मीटर के दायरे में यदि एक ही गड्ढे या कचरे की समस्या आती है, तो सिस्टम उन शिकायतों को क्लस्टर (एकत्र) कर लेता है। सभी नागरिकों को अलग-अलग प्रगति अपडेट मिलते हैं, लेकिन ठेकेदार को एक ही वर्क ऑर्डर दिया जाता है ताकि काम और बिल का दोहराव न हो।",
        "content_en": "If multiple citizens report the same civic issue within a 500-meter radius, the system clusters them. Every citizen receives individual status updates, while a single work order is issued to the contractor to prevent duplicate billing and effort."
    },

    # =========================================================================
    # 2. Municipal Department & Issue Classification Guide (महापालिका विभाग व तक्रार वर्गीकरण)
    # =========================================================================
    {
        "id": "dept_road_traffic",
        "keywords": [
            "road", "traffic", "pothole", "खड्डा", "खड्डे", "गड्ढा", "गड्ढे", "रस्ता", "सड़क", "डांबर", "डामर", "speed breaker", "divider", "nhai", "pwd"
        ],
        "title": "रस्ते आणि वाहतूक पायाभूत सुविधा विभाग (Road & Traffic Infrastructure)",
        "content_mr": "रस्ते आणि वाहतूक पायाभूत सुविधा विभाग:\n- कार्यक्षेत्र: रस्त्यावरील खड्डे (Potholes), उखडलेले डांबर, रस्त्यावरील चढ-उतार, तुटलेले फूटपाथ, पेव्हर ब्लॉक्स, दुभाजक (Dividers), अनधिकृत गतिरोधक, तुटलेले दिशादर्शक फलक.\n- महत्त्वाचे: राष्ट्रीय महामार्ग (NHAI) किंवा राज्य सार्वजनिक बांधकाम विभाग (PWD) चे उड्डाणपूल व रस्ते PWD नोडल अधिकाऱ्यांकडे पाठवले जातात.",
        "content_hi": "सड़क एवं यातायात अवसंरचना विभाग:\n- कार्यक्षेत्र: सड़क के गड्ढे (Potholes), उखड़ा हुआ डामर, असमान सड़क, टूटे फुटपाथ, पेवर ब्लॉक, डिवाइडर, अवैध स्पीड ब्रेकर, टूटे संकेत बोर्ड।\n- महत्वपूर्ण: राष्ट्रीय राजमार्ग (NHAI) या राज्य PWD के फ्लाईओवर व सड़कें संबंधित PWD नोडल अधिकारियों को अग्रेषित की जाती हैं।",
        "content_en": "Road & Traffic Infrastructure Department handles potholes, damaged asphalt, uneven road surfaces, broken footpaths, paver blocks, dividers, illegal speed breakers, and missing signages. National highways (NHAI) and state PWD flyovers are routed to designated PWD nodal officers."
    },
    {
        "id": "dept_solid_waste",
        "keywords": [
            "garbage", "kachra", "कचरा", "कूड़ा", "कचराकुंडी", "कूड़ेदान", "dustbin", "debris", "डेब्रिज", "मलबा", "घंटागाडी", "कचरा गाड़ी", "मृत जनावरे", "swachhata", "solid waste"
        ],
        "title": "घनकचरा व्यवस्थापन विभाग (Solid Waste Management)",
        "content_mr": "घनकचरा व्यवस्थापन विभाग:\n- समाविष्ट समस्या: कचराकुंडी भरून वाहणे (Overflowing Dustbins), रस्त्याच्या कडेला कचऱ्याचे उकिरडे, डेब्रिज (बांधकाम कचरा) टाकणे, घंटागाडी न येणे (Irregular Door-to-Door Collection), सार्वजनिक रस्त्यावरील मृत जनावरे हटवणे, भाजी मार्केटमधील कचरा साचणे.\n- तक्रार निवारण SLA: २४ तास.",
        "content_hi": "ठोस अपशिष्ट प्रबंधन विभाग:\n- सम्मिलित समस्याएं: कूड़ेदान भर जाना (Overflowing Dustbins), सड़क किनारे कचरा फेंकना, निर्माण मलबा (Debris), कचरा गाड़ी का न आना, मृत पशुओं को हटाना, सब्जी मंडी का कचरा।\n- शिकायत निवारण SLA: २४ घंटे।",
        "content_en": "Solid Waste Management Department handles overflowing garbage bins, illegal roadside dumps, construction debris, missed door-to-door collection vans (Ghantagadi), dead animal removal, and vegetable market waste clearance. Standard SLA: 24 hours."
    },
    {
        "id": "dept_water_supply",
        "keywords": [
            "water", "pani", "पाणी", "पानी", "पाणीपुरवठा", "जलापूर्ति", "pipe leak", "pipeline", "गळती", "लीकेज", "कमी दाब", "कम दबाव", "दूषित पाणी", "गंदा पानी", "valve", "water pressure"
        ],
        "title": "पाणीपुरवठा विभाग (Water Supply & Hydraulic Engineering)",
        "content_mr": "पाणीपुरवठा विभाग (Hydraulic Engineering):\n- समाविष्ट समस्या: सार्वजनिक रस्त्यावरील पिण्याच्या पाण्याच्या पाईपलाईनची गळती (Pipeline Leaks), पाणीपुरवठा बंद असणे किंवा कमी दाबाने पाणी येणे, नळाला दूषित किंवा गढूळ पाणी येणे, महापालिकेचे व्हॉल्व्ह (Valves) खराब असणे.\n- मुख्य पाईप गळती निवारण SLA: १२ ते २४ तास.",
        "content_hi": "जल आपूर्ति एवं हाइड्रोलिक विभाग:\n- सम्मिलित समस्याएं: सार्वजनिक सड़क पर पेयजल पाइपलाइन लीकेज, पानी की आपूर्ति बंद होना या कम दबाव (Low Pressure), नलों में गंदा/दूषित पानी आना, खराब म्यूनिसिपल वाल्व।\n- मुख्य पाइपलाइन लीकेज निवारण SLA: १२ से २४ घंटे।",
        "content_en": "Water Supply Department resolves public drinking water main pipeline bursts, supply disruptions, low water pressure, contaminated/muddy tap water, and damaged municipal sluice valves. SLA for main pipeline leak repair: 12 to 24 hours."
    },
    {
        "id": "dept_drainage_sewerage",
        "keywords": [
            "drainage", "sewer", "gutter", "gatar", "गटार", "नाली", "ड्रेनेज", "मॅनहोल", "मैनहोल", "manhole", "waterlogging", "जलभराव", "पाणी साचणे", "सांडपाणी", "सीवर"
        ],
        "title": "ड्रेनेज आणि मलनिस्सारण विभाग (Drainage & Sewerage - CRITICAL LIFE SAFETY)",
        "content_mr": "ड्रेनेज आणि मलनिस्सारण विभाग:\n- समाविष्ट समस्या: उघडे किंवा तुटलेले मॅनहोल झाकण (Open / Broken Manhole - अतिधोकादायक व तातडीचे), तुंबलेली गटारे, सांडपाणी रस्त्यावर वाहणे, चेंबर ब्लॉक होणे, पावसाळ्यात रस्त्यावर किंवा घरांमध्ये पाणी साचणे (Waterlogging), गटाराचे चेंबर खचणे.\n- SLA: २४ ते ४८ तास. उघड्या मॅनहोलवर तात्काळ कार्यवाही केली जाते.",
        "content_hi": "जल निकासी एवं सीवरेज विभाग:\n- सम्मिलित समस्याएं: खुला या टूटा हुआ मैनहोल ढक्कन (Open Manhole - अति-गंभीर व जीवन सुरक्षा आपातकाल), जाम नालियां, सड़क पर बहता सीवर, चेंबर ब्लॉकेज, बारिश में जलभराव (Waterlogging)।\n- निवारण SLA: २४ से ४८ घंटे (खुले मैनहोल पर आपातकालीन तत्काल कार्रवाई)।",
        "content_en": "Drainage & Sewerage Department handles open/broken manhole covers (CRITICAL LIFE SAFETY priority), choked sewers, overflowing gutters, storm waterlogging, and sunken chamber slabs. SLA: 24 to 48 hours with immediate emergency dispatch for open manholes."
    },
    {
        "id": "dept_electrical_streetlighting",
        "keywords": [
            "electricity", "streetlight", "light", "दिवा", "लाईट", "बिजली", "स्ट्रीट लाइट", "खांब", "खंभा", "pole", "wire", "msedcl", "1912", "current", "विद्युत", "sparking"
        ],
        "title": "विद्युत आणि पथदिवा विभाग (Electrical & Street Lighting)",
        "content_mr": "विद्युत आणि पथदिवा विभाग:\n- समाविष्ट समस्या: रात्री पथदिवा बंद असणे (Streetlight Off), दिवसा पथदिवा सुरू राहणे, विजेचा खांब वाकणे किंवा गंजणे, विजेच्या खांबावरील उघडी वायर पेटी (Exposed Junction Box).\n- महत्त्वाचे: हाय-टेन्शन ओव्हरहेड वायर स्पार्किंग, ट्रान्सफॉर्मर स्फोट किंवा मुख्य विद्युत तारा तुटल्यास महावितरण (MSEDCL) हेल्पलाइन १९१२ वर त्वरित संपर्क साधावा.",
        "content_hi": "विद्युत एवं पथप्रकाश (Street Light) विभाग:\n- सम्मिलित समस्याएं: रात में स्ट्रीट लाइट बंद होना, दिन में अनावश्यक लाइट जलना, झुका या सड़ा हुआ बिजली का खंभा, खंभे पर खुला जंक्शन बॉक्स।\n- महत्वपूर्ण: हाई-टेंशन ओवरहेड तार स्पार्किंग, ट्रांसफार्मर ब्लास्ट या मुख्य तार टूटने पर महावितरण (MSEDCL) हेल्पलाइन १९१२ पर तुरंत कॉल करें।",
        "content_en": "Electrical & Street Lighting Department manages non-functioning streetlights, day-burning lights, leaning/corroded poles, and exposed base junction boxes. Note: For high-tension live wire sparks or transformer explosions, contact MSEDCL helpline 1912 directly."
    },
    {
        "id": "dept_public_health_sanitation",
        "keywords": [
            "fogging", "sanitation", "mosquito", "डास", "मच्छर", "फवारणी", "छिड़काव", "dengue", "malaria", "डेंगू", "मलेरिया", "सार्वजनिक शौचालय", "दुर्गंधी", "सफाई"
        ],
        "title": "सार्वजनिक आरोग्य व स्वच्छता विभाग (Public Health & Sanitation)",
        "content_mr": "सार्वजनिक आरोग्य व स्वच्छता विभाग:\n- समाविष्ट समस्या: साचलेल्या पाण्यामुळे डासांची उत्पत्ती, मलेरिया/डेंग्यू धुरफवारणी (Fogging) न होणे, सार्वजनिक शौचालयांची अस्वच्छता, सार्वजनिक ठिकाणी दुर्गंधी व कीटक नियंत्रण.",
        "content_hi": "सार्वजनिक स्वास्थ्य एवं स्वच्छता विभाग:\n- सम्मिलित समस्याएं: रुके पानी से मच्छरों का प्रकोप, डेंगू/मलेरिया फॉगिंग (धुआं छिड़काव) न होना, सार्वजनिक शौचालयों की गंदगी, कीटनाशक छिड़काव व दुर्गंध नियंत्रण।",
        "content_en": "Public Health & Sanitation Department oversees vector mosquito control, dengue/malaria chemical fogging, public toilet sanitation, disinfection, and civic pest control."
    },
    {
        "id": "dept_garden_tree_authority",
        "keywords": [
            "tree", "garden", "झाड", "पेड़", "फांदी", "डालियां", "branch", "उद्यान", "बागा", "बगीचा", "धोकादायक झाड", "fallen tree"
        ],
        "title": "उद्यान आणि वृक्ष प्राधिकरण विभाग (Garden & Tree Authority)",
        "content_mr": "उद्यान आणि वृक्ष प्राधिकरण विभाग:\n- समाविष्ट समस्या: वादळामुळे किंवा पावसामुळे झाडाच्या फांद्या तुटून रस्त्यावर पडणे, विजेच्या तारांना किंवा दिव्यांना अडथळा ठरणारी धोकादायक झाडे, सार्वजनिक बागांची निगा, बागेतील तुटलेली खेळणी व बाकडे.",
        "content_hi": "उद्यान एवं वृक्ष प्राधिकरण विभाग:\n- सम्मिलित समस्याएं: आंधी/बारिश से पेड़ या टहनियां टूटकर सड़क पर गिरना, बिजली की तारों या बत्तियों में बाधक खतरनाक पेड़, सार्वजनिक पार्कों का रखरखाव, टूटे झूले व बेंच।",
        "content_en": "Garden & Tree Authority Department handles fallen storm branches blocking traffic, pruning hazardous branches near power lines, and maintaining public municipal parks and broken playground equipment."
    },

    # =========================================================================
    # 3. Grievance Escalation Process & Organizational Hierarchy (प्रशासकीय उतरंड)
    # =========================================================================
    {
        "id": "escalation_hierarchy",
        "keywords": [
            "escalation", "hierarchy", "उतरंड", "प्रशासनिक सोपान", "अधिकारी स्तर", "je", "ae", "ee", "ward officer", "councillor", "नगरसेवक", "पार्षद", "engineer", "level 1", "level 2"
        ],
        "title": "तक्रार निवारण प्रशासकीय उतरंड (4-Tier Escalation Ladder)",
        "content_mr": "तक्रार ठरलेल्या मुदतीत न सुटल्यास आपोआप (Automated Escalation) पुढील स्तरांवर जाते:\n- Level 1: क्षेत्रीय प्रत्यक्ष काम -> कनिष्ठ अभियंता (Junior Engineer - JE) / स्वच्छता निरीक्षक (Sanitary Inspector)\n- Level 2: पर्यवेक्षी स्तर -> सहाय्यक अभियंता (Assistant Engineer - AE)\n- Level 3: कार्यकारी पुनरावलोकन -> कार्यकारी अभियंता (Executive Engineer - EE - निधी वाटपाचे अधिकार)\n- Level 4: प्रभाग सर्वोच्च प्रमुख -> प्रभाग अधिकारी (Ward Officer / सहाय्यक आयुक्त) आणि स्थानिक नगरसेवक (Ward Councillor).",
        "content_hi": "शिकायत निर्धारित समयसीमा में न सुलझने पर स्वतः उच्च अधिकारियों को एस्केलेट (Automated Escalation) होती है:\n- Level 1 (मौके पर प्रत्यक्ष काम): कनिष्ठ अभियंता (JE) / स्वच्छता निरीक्षक\n- Level 2 (पर्यवेक्षण स्तर): सहायक अभियंता (AE)\n- Level 3 (कार्यकारी समीक्षा व बजट आवंटन): अधिशासी अभियंता (EE)\n- Level 4 (प्रभाग सर्वोच्च प्रमुख): वार्ड अधिकारी (सहायक आयुक्त) एवं स्थानीय पार्षद (Ward Councillor)।",
        "content_en": "Grievances unresolved within SLA automatically escalate through a 4-tier ladder:\n- Level 1 (Field Execution): Junior Engineer (JE) / Sanitary Inspector.\n- Level 2 (Supervisory): Assistant Engineer (AE).\n- Level 3 (Executive Review & Fund Allocation): Executive Engineer (EE).\n- Level 4 (Apex Ward Authority): Ward Officer (Assistant Commissioner) & Ward Councillor."
    },
    {
        "id": "reopen_complaint_ticket",
        "keywords": [
            "reopen", "reopen ticket", "पुन्हा उघडा", "शिकायत दोबारा खोलना", "काम झाले नाही", "कामाचे समाधान नाही", "निकृष्ट काम", "48 hours", "४८ तास"
        ],
        "title": "काम समाधानकारक न झाल्यास तक्रार पुन्हा उघडणे (Reopen Ticket)",
        "content_mr": "जर अधिकाऱ्याने तक्रार 'निवारण झाले' (Resolved) अशी बंद केली, पण प्रत्यक्षात जागेवर काम अर्धवट किंवा निकृष्ट असेल, तर नागरिक ४८ तासांच्या आत नवीन फोटो पाठवून तक्रार पुन्हा उघडू (Reopen) शकतात.",
        "content_hi": "यदि अधिकारी ने शिकायत 'निवारित' (Resolved) के रूप में बंद कर दी है, लेकिन मौके पर काम अधूरा या घटिया स्तर का है, तो नागरिक ४८ घंटे के भीतर नया फोटो भेजकर टिकट पुनः खोल (Reopen) सकते हैं।",
        "content_en": "If a grievance is marked 'Resolved' but the on-site work is incomplete or unsatisfactory, citizens can reopen the ticket within 48 hours by providing a fresh photo of the unresolved issue."
    },

    # =========================================================================
    # 4. Critical Emergency & Civic Life-Safety Helplines (आपत्कालीन संपर्क क्रमांक)
    # =========================================================================
    {
        "id": "emergency_helplines_national",
        "keywords": [
            "emergency", "helpline", "112", "101", "108", "100", "1912", "आपत्कालीन", "आपातकालीन", "अग्निशामक", "दमकल", "पोलीस", "पुलिस", "रुग्णवाहिका", "एम्बुलेंस", "महावितरण"
        ],
        "title": "राष्ट्रीय व महापालिका आपत्कालीन क्रमांक (Emergency Helplines - 24x7)",
        "content_mr": "२४ तास विनामूल्य आपत्कालीन संपर्क क्रमांक:\n- 112: सर्वसमावेशक राष्ट्रीय आपत्कालीन क्रमांक (पोलीस, अग्निशामक, रुग्णवाहिका).\n- 101: अग्निशामक दल (Fire Brigade - आग, गॅस सिलेंडर स्फोट).\n- 108: मोफत रुग्णवाहिका आणि वैद्यकीय आपत्कालीन सेवा (Ambulance).\n- 100 / 112: पोलीस नियंत्रण कक्ष (Police).\n- 1912: महावितरण / वीज वितरण आपत्कालीन सेवा (MSEDCL Live Wire, Transformer Sparking).",
        "content_hi": "२४ घंटे टोल-फ्री राष्ट्रीय आपातकालीन हेल्पलाइन नंबर:\n- 112: एकीकृत राष्ट्रीय आपातकालीन नंबर (पुलिस, दमकल, एम्बुलेंस)।\n- 101: दमकल / अग्निशमन दल (Fire Brigade - आग, गैस सिलेंडर रिसाव)।\n- 108: निःशुल्क एम्बुलेंस एवं आपातकालीन चिकित्सा सेवा।\n- 100 / 112: पुलिस नियंत्रण कक्ष (Police)।\n- 1912: महावितरण बिजली वितरण आपातकालीन सेवा (MSEDCL Live Wire, Transformer Sparks)।",
        "content_en": "24x7 Toll-Free Emergency Numbers:\n- 112: National Unified Emergency (Police, Fire, Medical).\n- 101: Fire Brigade (Fires, cylinder blasts, chemical hazards).\n- 108: Free Emergency Ambulance Medical Service.\n- 100 / 112: Police Control Room.\n- 1912: MSEDCL Power Discom Emergency (Live wires, transformer sparks)."
    },
    {
        "id": "emergency_safety_special",
        "keywords": [
            "women helpline", "childline", "elderline", "1091", "181", "1098", "14567", "1077", "महिला", "बाल", "ज्येष्ठ नागरिक", "वरिष्ठ नागरिक", "disaster cell", "आपत्ती कक्ष"
        ],
        "title": "विशेष नागरिक सुरक्षा व आपत्ती हेल्पलाइन्स (Special Civic Helplines)",
        "content_mr": "विशेष नागरिक सुरक्षा हेल्पलाइन्स:\n- 1091 / 181: महिला हेल्पलाईन (Women in Distress / Safety Helpline).\n- 1098: राष्ट्रीय बाल हेल्पलाईन (Childline).\n- 14567: ज्येष्ठ नागरिक हेल्पलाईन (Elderline).\n- 1077: जिल्हा आपत्ती व्यवस्थापन कक्ष (District Disaster Cell - पूर, दरड कोसळणे, इमारत पडणे).",
        "content_hi": "विशेष नागरिक सुरक्षा एवं आपदा हेल्पलाइन नंबर:\n- 1091 / 181: महिला सुरक्षा हेल्पलाइन (संकटग्रस्त महिलाओं हेतु)।\n- 1098: राष्ट्रीय चाइल्डलाइन (बच्चों की सहायता हेतु)।\n- 14567: एल्डरलाइन (वरिष्ठ नागरिकों की सहायता हेतु)।\n- 1077: जिला आपदा प्रबंधन नियंत्रण कक्ष (बाढ़, भूस्खलन, भवन ढहना)।",
        "content_en": "Specialized Citizen Safety Helplines:\n- 1091 / 181: Women in Distress & Safety Helpline.\n- 1098: National Childline Helpline.\n- 14567: Elderline Senior Citizen Helpline.\n- 1077: District Disaster Management Cell (Floods, landslides, structural collapses)."
    },
    {
        "id": "life_safety_critical_hazards",
        "keywords": [
            "hazard", "live wire", "gas leak", "collapse", "धोकादायक", "खतरनाक", "जिवंत तार", "बिजली का तार", "मॅनहोल उघडे", "खुला मैनहोल", "गॅस गळती", "गैस रिसाव", "इमारत पडणे"
        ],
        "title": "अति-धोकादायक परिस्थिती आणि तत्काळ कृती (Critical Hazard Guidelines)",
        "content_mr": "अति-धोकादायक परिस्थिती उद्भवल्यास त्वरित करावयाची कृती:\n- रस्त्यावर विजेची जिवंत तार तुटून पडल्यास: १० मीटर अंतर ठेवा आणि तात्काळ 1912 व 112 वर संपर्क साधा.\n- उघडे धोकादायक मॅनहोल: तात्काळ प्रभाग नियंत्रण कक्ष किंवा 112 वर संपर्क साधा.\n- गॅस गळती: 101 (अग्निशामक) वर कॉल करा आणि परिसर तात्काळ रिकामी करा.\n- इमारत खचणे किंवा भिंत कोसळणे: 112 आणि 1077 वर तात्काळ संपर्क करा.",
        "content_hi": "अति-गंभीर एवं जीवन सुरक्षा आपातकाल में तत्काल कदम:\n- सड़क पर बिजली का लाइव तार टूटकर गिरने पर: कम से कम १० मीटर की दूरी बनाएं और तुरंत 1912 व 112 पर कॉल करें।\n- खुला या टूटा खतरनाक मैनहोल: तुरंत 112 या वार्ड नियंत्रण कक्ष को सूचित करें।\n- रसोई गैस रिसाव (Gas Leak): तुरंत 101 (दमकल) पर कॉल करें और स्थान खाली करें।\n- इमारत या दीवार ढहना: तुरंत 112 और 1077 (आपदा कक्ष) पर सूचित करें।",
        "content_en": "Critical Life Safety Hazard Protocols:\n- Fallen live electric wires: Maintain minimum 10-meter distance, call 1912 and 112 immediately.\n- Open dangerous manholes: Immediately call 112 or Ward Control Room.\n- Gas leaks: Call 101 (Fire Brigade) and evacuate the area immediately.\n- Building/structural collapse: Call 112 and 1077 (Disaster Cell) immediately."
    },

    # =========================================================================
    # 5. Multilingual Civic Grievance Glossary & Designations (बोलीभाषा व पदे)
    # =========================================================================
    {
        "id": "civic_glossary_designations",
        "keywords": [
            "glossary", "terms", "नगरसेवक", "पार्षद", "प्रभाग अधिकारी", "वार्ड अधिकारी", "कनिष्ठ अभियंता", "cfc", "councillor", "ward officer", "je", "शब्दकोश", "boli bhasha", "shabdavali"
        ],
        "title": "प्रशासकीय पदे व बहुभाषिक नागरी शब्दकोश (Designations & Glossary)",
        "content_mr": "महापालिका प्रशासकीय पदे व शब्दकोश:\n- Ward Councillor: नगरसेवक / नगरसेविका (Parshad)\n- Ward Officer: प्रभाग अधिकारी / सहाय्यक आयुक्त (Ward Executive)\n- Junior Engineer: कनिष्ठ अभियंता (JE - जागेवर प्रत्यक्ष काम करणारे)\n- Citizen Facilitation Centre: नागरी सुविधा केंद्र (CFC).\nखड्डे (Pothole/Gaddha), कचरा (Garbage/Kooda), सांडपाणी/गटार (Gutter/Drainage), नळ गळती (Pipe Leak), पथदिवा (Streetlight).",
        "content_hi": "नगरपालिका प्रशासनिक पद एवं बहुभाषी शब्दावली:\n- Ward Councillor: पार्षद / नगरसेवक (स्थानीय निर्वाचित जनप्रतिनिधि)\n- Ward Officer: प्रभाग अधिकारी / सहायक आयुक्त (प्रशासनिक प्रमुख)\n- Junior Engineer (JE): कनिष्ठ अभियंता (फील्ड पर काम कराने वाले अभियंता)\n- Citizen Facilitation Centre: नागरिक सुविधा केंद्र (CFC)।\nगड्ढे (Pothole/Khadda), कचरा (Garbage/Kachra), नाली/सीवर (Gutter/Drainage), पाइप लीकेज (Water Leak), स्ट्रीट लाइट (पथदिवा)।",
        "content_en": "Municipal Designations & Vocabulary Guide:\n- Ward Councillor: नगरसेवक / नगरसेविका (Parshad / Local Elected Representative)\n- Ward Officer: प्रभाग अधिकारी / सहाय्यक आयुक्त (Ward Executive)\n- Junior Engineer (JE): कनिष्ठ अभियंता (Field Engineer)\n- Citizen Facilitation Centre (CFC): नागरी सुविधा केंद्र.\nPotholes (खड्डे / गड्ढे), Garbage (कचरा / कूड़ा), Gutter (गटार / नाली), Pipe Leak (पाईप गळती), Streetlight (पथदिवा)."
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
