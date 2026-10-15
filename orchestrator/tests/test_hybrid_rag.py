import pytest
from app.tools.hybrid_rag import tokenize_multilingual, BM25Scorer, HybridMunicipalKnowledgeBase
from app.tools.rag_knowledge import MunicipalKnowledgeBase, MUNICIPAL_KNOWLEDGE_DOCS

def test_tokenize_multilingual():
    text = "महापालिका कार्यालयीन वेळ काय आहे? Help 1912 & ward 53"
    tokens = tokenize_multilingual(text)
    assert "महापालिका" in tokens
    assert "कार्यालयीन" in tokens
    assert "वेळ" in tokens
    assert "help" in tokens
    assert "1912" in tokens
    assert "53" in tokens

def test_bm25_scorer_marathi_query():
    scorer = BM25Scorer(MUNICIPAL_KNOWLEDGE_DOCS)
    # Query for office timings in Marathi
    scores = scorer.score("कार्यालयीन वेळ कधी आहे आणि कधी उघडते?")
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    top_doc = MUNICIPAL_KNOWLEDGE_DOCS[top_doc_idx]
    assert top_doc["id"] == "office_timings"

def test_bm25_scorer_emergency_helpline():
    scorer = BM25Scorer(MUNICIPAL_KNOWLEDGE_DOCS)
    scores = scorer.score("आपत्कालीन फोन नंबर आणि फायर ब्रिगेड")
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    top_doc = MUNICIPAL_KNOWLEDGE_DOCS[top_doc_idx]
    assert top_doc["id"] == "emergency_helpline"

def test_bm25_scorer_sla_garbage():
    scorer = BM25Scorer(MUNICIPAL_KNOWLEDGE_DOCS)
    scores = scorer.score("कचरा उचलण्यासाठी किती दिवस किंवा वेळ लागेल?")
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    top_doc = MUNICIPAL_KNOWLEDGE_DOCS[top_doc_idx]
    assert top_doc["id"] == "grievance_sla"

def test_hybrid_knowledge_base_search():
    kb = HybridMunicipalKnowledgeBase(docs=MUNICIPAL_KNOWLEDGE_DOCS)
    res = kb.search("कार्यालय कधी बंद होते?")
    assert res["found"] is True
    assert len(res["results"]) > 0
    assert "१०:०० ते सायंकाळी ५:४५" in res["results"][0]["content_mr"]

def test_hybrid_knowledge_base_out_of_scope():
    kb = HybridMunicipalKnowledgeBase(docs=MUNICIPAL_KNOWLEDGE_DOCS)
    res = kb.search("space rocket launch to mars")
    assert res["found"] is False
    assert "१८००-२३३-००५०" in res["message"]

def test_municipal_knowledge_base_wrapper():
    kb = MunicipalKnowledgeBase()
    res = kb.search("पाणीपुरवठा आणि गळती दुरुस्ती वेळ")
    assert res["found"] is True
    assert len(res["results"]) > 0
