import pytest

from ingest.chunk import chunk_sections
from ingest.parse import Section


def make_section(
    text: str, doc_id: str = "doc1", section_path: str = "Doc > H1", anchor: str = "h1"
) -> Section:
    return Section(
        doc_id=doc_id,
        section_path=section_path,
        heading_level=1,
        anchor=anchor,
        text=text,
        source_url="https://example.com/doc1",
    )


def test_short_section_produces_a_single_chunk():
    section = make_section("A short paragraph about syphilis treatment.")
    chunks = chunk_sections([section], max_tokens=700, overlap_tokens=120)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text.startswith("Doc > H1\n\n")
    assert "syphilis" in chunks[0].text


def test_oversized_section_splits_into_multiple_overlapping_chunks():
    # ~2000 words guarantees this exceeds a 100-token budget many times over.
    long_text = " ".join(f"word{i}" for i in range(2000))
    section = make_section(long_text)

    chunks = chunk_sections([section], max_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= 105  # small slack for prefix/decoding rounding


def test_overlap_shares_tokens_between_consecutive_chunks():
    long_text = " ".join(f"word{i}" for i in range(500))
    section = make_section(long_text)

    chunks = chunk_sections([section], max_tokens=100, overlap_tokens=20)

    first_words = chunks[0].text.split("\n\n", 1)[1].split()
    second_words = chunks[1].text.split("\n\n", 1)[1].split()
    overlap = set(first_words[-15:]) & set(second_words[:15])
    assert overlap, "expected shared tokens between consecutive overlapping windows"


def test_every_chunk_carries_metadata():
    section = make_section(
        "Some content.", doc_id="sti-syphilis", section_path="Syphilis > Treatment", anchor="tx"
    )
    chunks = chunk_sections([section])

    chunk = chunks[0]
    assert chunk.doc_id == "sti-syphilis"
    assert chunk.section_path == "Syphilis > Treatment"
    assert chunk.anchor == "tx"
    assert chunk.source_url == "https://example.com/doc1"
    assert chunk.chunk_id.startswith("sti-syphilis::")


def test_chunking_is_deterministic():
    sections = [make_section("Repeatable content for determinism check.")]

    result_a = chunk_sections(sections)
    result_b = chunk_sections(sections)

    assert result_a == result_b


def test_chunk_ids_are_unique_across_sections_and_indices():
    sections = [
        make_section("First section content.", section_path="Doc > A", anchor="a"),
        make_section("Second section content.", section_path="Doc > B", anchor="b"),
    ]
    chunks = chunk_sections(sections)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_sections_list_produces_no_chunks():
    assert chunk_sections([]) == []


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        chunk_sections([make_section("text")], max_tokens=50, overlap_tokens=50)
