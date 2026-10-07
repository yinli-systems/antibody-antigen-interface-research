from ingest.chunking import split_parent_child


def test_parent_child_offsets():
    text = " ".join(["residue"] * 200)
    parts = split_parent_child("p1", "t", text, child_words=80)
    assert len(parts) >= 2
    cid, span, start, end = parts[0]
    assert span in text
    assert start == 0
