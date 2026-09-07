import json

import httpx

from ingest.fetch import fetch_all
from ingest.sources import SourceDoc

DOCS = [
    SourceDoc("doc-a", "Doc A", "https://example.com/a.htm"),
    SourceDoc("doc-b", "Doc B", "https://example.com/b.htm"),
]


def handler(request: httpx.Request) -> httpx.Response:
    body = f"<html><body><h1>{request.url.path}</h1></body></html>"
    return httpx.Response(200, text=body)


def make_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_all_writes_one_html_file_per_doc(tmp_path):
    with make_client() as client:
        records = fetch_all(tmp_path, client, docs=DOCS)

    assert len(records) == 2
    assert (tmp_path / "doc-a.html").exists()
    assert (tmp_path / "doc-b.html").exists()


def test_fetch_all_writes_a_manifest_json(tmp_path):
    with make_client() as client:
        fetch_all(tmp_path, client, docs=DOCS)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    doc_ids = {entry["doc_id"] for entry in manifest}
    assert doc_ids == {"doc-a", "doc-b"}
    assert all(entry["sha256"] for entry in manifest)


def test_fetch_all_skips_existing_files_unless_forced(tmp_path):
    calls = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text="<html><body>content</body></html>")

    with httpx.Client(transport=httpx.MockTransport(counting_handler)) as client:
        fetch_all(tmp_path, client, docs=DOCS[:1])
        fetch_all(tmp_path, client, docs=DOCS[:1])  # second call should not re-fetch

    assert len(calls) == 1


def test_force_refetches_even_if_cached(tmp_path):
    calls = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text="<html><body>content</body></html>")

    with httpx.Client(transport=httpx.MockTransport(counting_handler)) as client:
        fetch_all(tmp_path, client, docs=DOCS[:1])
        fetch_all(tmp_path, client, docs=DOCS[:1], force=True)

    assert len(calls) == 2


def test_raises_on_http_error(tmp_path):
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    with httpx.Client(transport=httpx.MockTransport(error_handler)) as client:
        try:
            fetch_all(tmp_path, client, docs=DOCS[:1])
            raised = False
        except httpx.HTTPStatusError:
            raised = True

    assert raised
