from agentlab.rag.hybrid import rrf


def test_rrf_prefers_items_found_by_both_methods():
    a = [{"doc_id": "1", "chunk_index": 0}, {"doc_id": "2", "chunk_index": 0}]
    b = [{"doc_id": "3", "chunk_index": 0}, {"doc_id": "2", "chunk_index": 0}]
    assert rrf([a, b], top_k=3)[0]["doc_id"] == "2"