"""Retrieval metadata must survive chunking and the vector index."""

from src.data import load_documents
from src.retrieval import chunk_documents


def test_chunks_keep_document_id_title_position_and_text(settings):
    chunks = chunk_documents(load_documents(settings.knowledge_base_dir))
    shipping = [c for c in chunks if c.document_id == "shipping"]
    assert [c.position for c in shipping] == list(range(len(shipping)))
    assert all(c.title == "Shipping Policy" and c.text for c in shipping)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_search_hits_carry_source_metadata_and_scores(retriever):
    hits = retriever.search("What is the usual standard shipping time?")
    assert hits[0].document_id == "shipping"
    assert hits[0].title == "Shipping Policy"
    assert hits[0].chunk_id.startswith("shipping-")
    assert hits[0].score >= hits[-1].score  # best first
    assert len(hits) == retriever.settings.top_k


def test_off_topic_question_has_no_evidence_above_threshold(retriever):
    hits, evidence = retriever.evidence_for("Where can I find next week's cafeteria menu?")
    assert hits  # the index always returns *something* ...
    assert evidence == []  # ... but nothing strong enough to count as evidence
