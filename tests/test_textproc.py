from narratortool.textproc import chunk, normalize, split_sentences


class TestNormalize:
    def test_rejoins_pdf_hyphenation(self):
        assert "narration" in normalize("The narra-\ntion was clear.")

    def test_drops_bare_page_numbers(self):
        out = normalize("End of thought.\n\n42\n\nNew thought begins.")
        assert "42" not in out

    def test_strips_gutenberg_boilerplate(self):
        raw = (
            "Legal preamble nobody wants read aloud.\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK 10 ***\n"
            "The actual book.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK 10 ***\n"
            "Licensing trailer."
        )
        out = normalize(raw)
        assert "The actual book." in out
        assert "preamble" not in out and "Licensing" not in out

    def test_boilerplate_kept_when_asked(self):
        raw = "Preamble.\n*** START OF THE PROJECT GUTENBERG EBOOK 10 ***\nBody."
        assert "Preamble" in normalize(raw, strip_boilerplate=False)

    def test_collapses_dot_leaders(self):
        assert "...." not in normalize("Chapter One......... 14")

    def test_preserves_paragraph_breaks(self):
        assert "\n\n" in normalize("First para.\n\nSecond para.")


class TestSplitSentences:
    def test_basic(self):
        assert len(split_sentences("One. Two. Three.")) == 3

    def test_does_not_split_on_titles(self):
        # "Dr." must not end a sentence, or the narrator pauses mid-name.
        assert len(split_sentences("Dr. Who arrived. Then left.")) == 2

    def test_does_not_split_on_eg(self):
        assert len(split_sentences("Use tools, e.g. a hammer. Then stop.")) == 2

    def test_handles_quotes(self):
        assert len(split_sentences('"Stop!" she cried. He did not.')) == 2


class TestChunk:
    def test_respects_max_chars(self):
        text = " ".join(f"Sentence number {i} here." for i in range(80))
        assert all(len(c) <= 120 for c in chunk(text, max_chars=120))

    def test_packs_rather_than_one_per_sentence(self):
        text = "A short one. Another short one. A third short one."
        assert len(chunk(text, max_chars=200)) == 1

    def test_never_merges_across_paragraphs(self):
        assert len(chunk("First para.\n\nSecond para.", max_chars=500)) == 2

    def test_splits_sentence_with_no_terminal_punctuation(self):
        chunks = chunk("word " * 300, max_chars=100)
        assert chunks and all(len(c) <= 100 for c in chunks)

    def test_loses_no_words(self):
        text = " ".join(f"Word{i}." for i in range(200))
        joined = " ".join(chunk(text, max_chars=90))
        assert joined.split() == text.split()

    def test_empty_input(self):
        assert chunk("") == []

    def test_rejects_bad_max(self):
        import pytest

        with pytest.raises(ValueError):
            chunk("text", max_chars=0)
