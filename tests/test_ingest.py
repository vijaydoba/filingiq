from app.ingest import chunk_text


def test_chunk_text_preserves_overlap_and_covers_input():
    words = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(words, chunk_size=10, overlap=3)

    assert chunks
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[0].split()[0] == "word0"
    assert chunks[-1].split()[-1] == "word29"


def test_chunk_text_empty_input():
    assert chunk_text("", chunk_size=10, overlap=3) == []
