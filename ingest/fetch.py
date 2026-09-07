import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ingest.sources import MANIFEST, SourceDoc

USER_AGENT = "clinical-rag-agent/0.1 (portfolio project; contact via GitHub)"


@dataclass(frozen=True)
class FetchRecord:
    doc_id: str
    url: str
    title: str
    fetched_at: str
    sha256: str
    path: str


def fetch_all(
    raw_dir: Path,
    client: httpx.Client,
    docs: list[SourceDoc] | None = None,
    force: bool = False,
) -> list[FetchRecord]:
    """Fetch each manifest doc to raw_dir/<doc_id>.html, skipping ones already cached
    unless force=True. Writes raw_dir/manifest.json summarizing what was fetched."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs = docs if docs is not None else MANIFEST
    records: list[FetchRecord] = []

    for doc in docs:
        dest = raw_dir / f"{doc.doc_id}.html"
        if dest.exists() and not force:
            html = dest.read_text(encoding="utf-8")
            records.append(_record_for(doc, dest, html, fetched_at=None))
            continue

        response = client.get(doc.url, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        response.raise_for_status()
        html = response.text
        dest.write_text(html, encoding="utf-8")
        records.append(_record_for(doc, dest, html, fetched_at=_now()))

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2),
        encoding="utf-8",
    )
    return records


def _record_for(doc: SourceDoc, dest: Path, html: str, fetched_at: str | None) -> FetchRecord:
    return FetchRecord(
        doc_id=doc.doc_id,
        url=doc.url,
        title=doc.title,
        fetched_at=fetched_at or _mtime_iso(dest),
        sha256=hashlib.sha256(html.encode("utf-8")).hexdigest(),
        path=str(dest),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


if __name__ == "__main__":
    with httpx.Client(timeout=30.0) as http_client:
        fetched = fetch_all(Path("ingest/raw"), http_client)
    print(f"Fetched/cached {len(fetched)} documents into ingest/raw/")
