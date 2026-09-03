import pytest

from narratortool.document import Chapter, Document
from narratortool.extract import UnsupportedFormat, extract, supported_extensions
from narratortool.extract.html import html_to_text


class TestDispatch:
    def test_unsupported_extension(self, tmp_path):
        bad = tmp_path / "audio.flac"
        bad.write_text("x")
        with pytest.raises(UnsupportedFormat):
            extract(bad)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract(tmp_path / "nope.txt")

    def test_registry_covers_documented_formats(self):
        for ext in (".txt", ".md", ".pdf", ".epub", ".docx", ".html"):
            assert ext in supported_extensions()


class TestPlaintext:
    def test_title_falls_back_to_filename(self, tmp_path):
        f = tmp_path / "My Book.txt"
        f.write_text("Just some prose.")
        assert extract(f).title == "My Book"

    def test_splits_on_chapter_headings(self, tmp_path):
        f = tmp_path / "novel.txt"
        f.write_text("Chapter One\nIt began.\n\nChapter Two\nIt continued.\n")
        assert len(extract(f).chapters) == 2

    def test_single_chapter_when_unstructured(self, tmp_path):
        f = tmp_path / "flat.txt"
        f.write_text("No headings at all, just text that runs on.")
        assert len(extract(f).chapters) == 1

    def test_handles_non_utf8(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes("caf\xe9 na\xefve".encode("latin-1"))
        assert "caf" in extract(f).text


class TestMarkdown:
    def test_headings_become_chapters(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# One\nAlpha text.\n\n# Two\nBeta text.\n")
        doc = extract(f)
        assert [c.title for c in doc.chapters] == ["One", "Two"]

    def test_strips_markup(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# T\nSome **bold** and a [link](http://x.com) and `code`.\n")
        text = extract(f).text
        assert "**" not in text and "http://x.com" not in text
        assert "bold" in text and "link" in text and "code" in text


class TestHTML:
    def test_drops_script_and_style(self):
        out = html_to_text("<p>Keep this.</p><script>bad()</script><style>p{}</style>")
        assert "Keep this." in out
        assert "bad()" not in out and "p{}" not in out

    def test_unescapes_entities(self):
        assert "&amp;" not in html_to_text("<p>Tom &amp; Jerry</p>")

    def test_block_tags_become_breaks(self):
        out = html_to_text("<p>One</p><p>Two</p>")
        assert "OneTwo" not in out


class TestDocumentModel:
    def test_drop_empty_removes_blank_chapters(self):
        doc = Document(chapters=[Chapter(text="real"), Chapter(text="   ")])
        assert len(doc.drop_empty().chapters) == 1

    def test_text_joins_chapters(self):
        doc = Document(chapters=[Chapter(text="A"), Chapter(text="B")])
        assert doc.text == "A\n\nB"

    def test_char_count(self):
        assert Document(chapters=[Chapter(text="12345")]).char_count == 5


class TestHeadingDetectionRegression:
    """Guards the bug where any wrapped prose line starting with a heading keyword
    became a chapter — e.g. 'book of the law, that he rent his clothes.'"""

    def test_ignores_wrapped_prose_starting_with_keyword(self, tmp_path):
        f = tmp_path / "wrapped.txt"
        f.write_text(
            "And it came to pass that he opened the\n"
            "book of the law, that he rent his clothes,\n"
            "and he wept before the people of the land.\n"
        )
        doc = extract(f)
        assert len(doc.chapters) == 1
        assert doc.chapters[0].title is None

    def test_accepts_standalone_headings(self, tmp_path):
        f = tmp_path / "real.txt"
        f.write_text(
            "Front matter line.\n\n"
            "Chapter One\n"
            "It began on a cold morning.\n\n"
            "Chapter Two\n"
            "It continued much later.\n"
        )
        titles = [c.title for c in extract(f).chapters if c.title]
        assert titles == ["Chapter One", "Chapter Two"]

    def test_rejects_heading_ending_in_comma(self, tmp_path):
        f = tmp_path / "comma.txt"
        f.write_text("Intro.\n\npart of the whole,\nmore text here.\n\npart two,\nand more.\n")
        assert len(extract(f).chapters) == 1


class _StubPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _StubReader:
    """A PdfReader stand-in: pypdf can parse PDFs but not author text-bearing ones,
    and a real fixture would mean a new build dependency for one assembly rule."""

    def __init__(self, pages: list[str], outline: list[tuple[str, int]]) -> None:
        self.pages = [_StubPage(t) for t in pages]
        self.metadata: dict[str, str] = {}
        self._outline = outline

    @property
    def outline(self):
        return [type("Item", (), {"title": t})() for t, _ in self._outline]

    def get_destination_page_number(self, item) -> int:
        return dict(self._outline)[item.title]


class TestPdfFrontMatter:
    """Guards the bug where everything ahead of the outline's first bookmark was
    dropped: a paper's title, author and abstract, narrated starting mid-sentence."""

    @staticmethod
    def _extract(monkeypatch, tmp_path, pages, outline):
        import pypdf

        monkeypatch.setattr(pypdf, "PdfReader", lambda _: _StubReader(pages, outline))
        f = tmp_path / "paper.pdf"
        f.write_bytes(b"%PDF-1.4")
        return extract(f)

    def test_keeps_pages_before_the_first_bookmark(self, monkeypatch, tmp_path):
        doc = self._extract(
            monkeypatch, tmp_path,
            ["Title Page. Abstract begins.", "Intro body.", "Method body."],
            [("Introduction", 1), ("Method", 2)],
        )
        assert "Title Page" in doc.text
        assert "Abstract begins" in doc.text

    def test_front_matter_leads_and_is_untitled(self, monkeypatch, tmp_path):
        doc = self._extract(
            monkeypatch, tmp_path,
            ["Front.", "Intro body.", "Method body."],
            [("Introduction", 1), ("Method", 2)],
        )
        assert [c.title for c in doc.chapters] == [None, "Introduction", "Method"]
        assert doc.chapters[0].text == "Front."

    def test_no_phantom_chapter_when_outline_starts_at_page_one(self, monkeypatch, tmp_path):
        doc = self._extract(
            monkeypatch, tmp_path,
            ["Intro body.", "Method body."],
            [("Introduction", 0), ("Method", 1)],
        )
        assert [c.title for c in doc.chapters] == ["Introduction", "Method"]

    def test_blank_front_matter_adds_no_chapter(self, monkeypatch, tmp_path):
        doc = self._extract(
            monkeypatch, tmp_path,
            ["   ", "Intro body.", "Method body."],
            [("Introduction", 1), ("Method", 2)],
        )
        assert [c.title for c in doc.chapters] == ["Introduction", "Method"]
