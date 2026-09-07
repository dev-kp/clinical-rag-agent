from ingest.parse import Section, parse_html

SAMPLE_HTML = """
<html><body>
<main>
<h1>Syphilis</h1>
<p>Intro paragraph about syphilis in general.</p>
<h2 id="treatment">Treatment</h2>
<p>Penicillin G is the preferred treatment.</p>
<h3 id="dosing">Dosing</h3>
<p>2.4 million units IM in a single dose.</p>
<h2 id="pregnancy">Syphilis During Pregnancy</h2>
<p>Pregnant patients should be treated with a penicillin regimen appropriate to stage.</p>
<script>console.log("ignore me")</script>
</main>
</body></html>
"""

DOC_ID = "sti-syphilis"
SOURCE_URL = "https://example.com/syphilis"
TITLE = "Syphilis"


def parse_sample() -> list[Section]:
    return parse_html(SAMPLE_HTML, doc_id=DOC_ID, source_url=SOURCE_URL, title=TITLE)


def test_parses_top_level_and_nested_headings():
    paths = [s.section_path for s in parse_sample()]

    assert "Syphilis" in paths
    assert "Syphilis > Treatment" in paths
    assert "Syphilis > Treatment > Dosing" in paths
    assert "Syphilis > Syphilis During Pregnancy" in paths


def test_section_text_only_contains_its_own_content():
    sections = parse_sample()

    dosing = next(s for s in sections if s.section_path == "Syphilis > Treatment > Dosing")
    assert "2.4 million units" in dosing.text
    assert "Penicillin G is the preferred" not in dosing.text


def test_script_tags_are_excluded():
    all_text = " ".join(s.text for s in parse_sample())
    assert "ignore me" not in all_text


def test_anchor_prefers_heading_id_attribute():
    sections = parse_sample()

    treatment = next(s for s in sections if s.section_path == "Syphilis > Treatment")
    assert treatment.anchor == "treatment"


def test_returning_to_a_shallower_heading_resets_breadcrumb():
    sections = parse_sample()

    pregnancy = next(s for s in sections if "Pregnancy" in s.section_path)
    assert pregnancy.section_path == "Syphilis > Syphilis During Pregnancy"
    assert "Dosing" not in pregnancy.section_path


def test_section_body_does_not_repeat_its_own_heading_text():
    sections = parse_sample()

    treatment = next(s for s in sections if s.section_path == "Syphilis > Treatment")
    assert treatment.text.count("Treatment") == 0


def test_source_url_and_doc_id_propagate_to_every_section():
    sections = parse_sample()

    assert all(s.doc_id == DOC_ID for s in sections)
    assert all(s.source_url == SOURCE_URL for s in sections)
