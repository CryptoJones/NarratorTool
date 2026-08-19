"""Format detection and dispatch.

Extractors are registered by file extension. Each one imports its heavy third-party
dependency lazily, inside the function — so a user who only ever narrates .txt files
never needs pypdf, ebooklib, or python-docx installed.
"""
from __future__ import annotations

from pathlib import Path

from ..document import Document

# extension -> "module:function" inside this package, resolved on demand.
_REGISTRY: dict[str, str] = {
    ".txt": "plaintext:extract_txt",
    ".text": "plaintext:extract_txt",
    ".md": "plaintext:extract_markdown",
    ".markdown": "plaintext:extract_markdown",
    ".pdf": "pdf:extract_pdf",
    ".epub": "epub:extract_epub",
    ".docx": "docx:extract_docx",
    ".html": "html:extract_html",
    ".htm": "html:extract_html",
}


class UnsupportedFormat(Exception):
    pass


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY)


def extract(path: str | Path) -> Document:
    """Parse `path` into a Document, choosing the extractor by extension."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    target = _REGISTRY.get(path.suffix.lower())
    if target is None:
        raise UnsupportedFormat(
            f"no extractor for {path.suffix!r}; supported: {', '.join(supported_extensions())}"
        )

    module_name, func_name = target.split(":")
    module = __import__(f"{__name__}.{module_name}", fromlist=[func_name])
    doc: Document = getattr(module, func_name)(path)
    doc.source_path = str(path)
    if not doc.title:
        doc.title = path.stem
    return doc.drop_empty()
