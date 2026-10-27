from agentlab.rag.chunking import chunk_fixed_size, chunk_sentence_based


def test_sentence_chunks_respect_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_sentence_based(text, max_size=200)
    assert all(len(c) <= 200 for c in chunks)


def test_fixed_size_overlap():
    assert len(chunk_fixed_size("a" * 1000, size=500, overlap=50)) == 3