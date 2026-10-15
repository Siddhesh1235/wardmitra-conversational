import pytest
from app.tools.hybrid_rag import tokenize_multilingual, BM25Scorer, HybridMunicipalKnowledgeBase
from app.tools.rag_knowledge import MunicipalKnowledgeBase, MUNICIPAL_KNOWLEDGE_DOCS

def test_tokenize_multilingual():
    text = "वॉर्डमित्र तक्रार नोंदणी कशी करायची? Help 1912 & ward 53"
    tokens = tokenize_multilingual(text)
    assert "वॉर्डमित्र" in tokens
    assert "तक्रार" in tokens
    assert "नोंदणी" in tokens
    assert "help" in tokens
    assert "1912" in tokens
    assert "53" in tokens

def test_bm25_scorer_marathi_query():
    scorer = BM25Scorer(MUNICIPAL_KNOWLEDGE_DOCS)
    # Query for filing complaint channels in Marathi
    scores = scorer.score("वॉर्डमित्र द्वारे तक्रार कशी नोंदवायची आणि व्हॉट्सॲप बॉट")
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    top_doc = MUNICIPAL_KNOWLEDGE_DOCS[top_doc_idx]
    assert top_doc["id"] == "how_to_file_complaint"

def test_bm25_scorer_hindi_query():
    scorer = BM25Scorer(MUNICIPAL_KNOWLEDGE_DOCS)
    # Query for solid waste / garbage in Hindi
    scores = scorer.score("सड़क पर कचरा और कूड़ेदान भर गया है सफाई कितने समय में होगी?")
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    top_doc = MUNICIPAL_KNOWLEDGE_DOCS[top_doc_idx]
    assert top_doc["id"] == "dept_solid_waste"

def test_bm25_scorer_emergency_helpline():
    scorer = BM25Scorer(MUNICIPAL_KNOWLEDGE_DOCS)
    scores = scorer.score("आपत्कालीन फोन नंबर आणि फायर ब्रिगेड 112")
    assert len(scores) > 0
    top_doc_idx = scores[0][0]
    top_doc = MUNICIPAL_KNOWLEDGE_DOCS[top_doc_idx]
    assert top_doc["id"] == "emergency_helplines_national"

def test_hybrid_knowledge_base_search():
    kb = HybridMunicipalKnowledgeBase(docs=MUNICIPAL_KNOWLEDGE_DOCS)
    res = kb.search("पिण्याच्या पाण्याची पाईपलाईन गळती")
    assert res["found"] is True
    assert len(res["results"]) > 0
    top = res["results"][0]
    assert top["id"] == "dept_water_supply"
    assert "content_mr" in top
    assert "content_hi" in top
    assert "content_en" in top

def test_hybrid_knowledge_base_out_of_scope():
    kb = HybridMunicipalKnowledgeBase(docs=MUNICIPAL_KNOWLEDGE_DOCS)
    res = kb.search("space rocket launch to mars")
    assert res["found"] is False
    assert "१८००-२३३-००५०" in res["message"]

def test_municipal_knowledge_base_wrapper():
    kb = MunicipalKnowledgeBase()
    res = kb.search("रस्त्यावरील खड्डे आणि डांबर")
    assert res["found"] is True
    assert len(res["results"]) > 0
    assert res["results"][0]["id"] == "dept_road_traffic"
