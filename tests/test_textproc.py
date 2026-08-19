from narratortool.textproc import chunk, find_figure_lines, normalize, split_sentences


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


class TestFigureSuppression:
    """PDF text layers include everything drawn on the page, charts included."""

    PAGE = """The model was trained for forty epochs on the full corpus.

Figure 3.2: Accuracy versus epochs for the three model variants.
Accuracy
1.0
0.8
0.6
0.4
0.2
0
10 20 30 40
Epochs
baseline ours ablation

As the plot shows, the ablation trails the baseline throughout training."""

    def test_caption_and_axis_text_are_dropped(self):
        out = normalize(self.PAGE)
        assert "Figure 3.2" not in out
        assert "0.8" not in out
        assert "10 20 30 40" not in out

    def test_surrounding_prose_survives(self):
        out = normalize(self.PAGE)
        assert "trained for forty epochs" in out
        assert "the ablation trails the baseline" in out

    def test_keeping_figures_is_opt_in(self):
        out = normalize(self.PAGE, strip_figures=False)
        assert "Figure 3.2" in out and "0.8" in out

    def test_find_figure_lines_reports_what_would_go(self):
        dropped = find_figure_lines(self.PAGE)
        assert any(d.startswith("Figure 3.2") for d in dropped)
        assert "10 20 30 40" in dropped
        assert not any("ablation trails" in d for d in dropped)

    def test_caption_label_forms(self):
        for caption in (
            "Figure 3.2: Accuracy versus epochs.",
            "Fig. 4 — Results for the three models.",
            "Table 2. Summary statistics.",
            "Figure 3.2",
            "Algorithm 2 Gradient descent with momentum",
            "Plate IV",
        ):
            assert find_figure_lines(f"Prose above.\n{caption}\nProse below."), caption

    def test_prose_mentioning_a_figure_is_not_a_caption(self):
        for prose in (
            "Figure 3 shows that accuracy improves with depth.",
            "Figures are shown throughout the chapter.",
            "Table of Contents",
            "The table 3 lines below is wrong.",
        ):
            assert not find_figure_lines(prose), prose

    def test_wrapped_caption_is_followed_to_its_end(self):
        text = (
            "Real prose here.\n\n"
            "Figure 5: Throughput measured across the three\n"
            "clusters over a single week of production traffic.\n\n"
            "More real prose."
        )
        out = normalize(text)
        assert "clusters over a single week" not in out
        assert "Real prose here." in out and "More real prose." in out

    def test_a_caption_does_not_swallow_the_following_paragraph(self):
        text = (
            "Figure 5: Throughput.\n"
            "This sentence is ordinary prose that follows the caption.\n"
            "So is this one, and it must survive."
        )
        out = normalize(text)
        assert "ordinary prose that follows" in out
        assert "it must survive" in out

    def test_prose_with_numbers_is_kept(self):
        for line in (
            "In 1901 he sailed for two years.",
            "January 1, 1901",
            "He was 6 feet 2 inches tall.",
            "Page 3 of 12",
            "2 cups flour",
        ):
            assert not find_figure_lines(line), line

    def test_headings_and_numerals_are_kept(self):
        for line in ("Introduction", "I.", "II", "Chapter One"):
            assert not find_figure_lines(line), line

    def test_long_lines_are_never_debris(self):
        line = "0.2 " * 30  # 120 chars of numbers, but too long to be an axis label
        assert not find_figure_lines(line.strip())

    def test_suppression_does_not_split_a_paragraph(self):
        """A caption floating inside a hard-wrapped paragraph must not break it in two."""
        text = (
            "The measurements were taken over a single week and\n"
            "Figure 5: Throughput.\n"
            "showed no meaningful variation between the clusters."
        )
        out = normalize(text)
        assert "single week and showed no meaningful variation" in out
        assert "\n" not in out


class TestChartLabelSweep:
    """Axis names and legend keys carry no digits — only their position gives them away."""

    def test_axis_names_beside_ticks_are_dropped(self):
        text = (
            "Real prose introducing the plot.\n"
            "Accuracy\n"
            "1.0\n"
            "0.8\n"
            "0.6\n"
            "0 10 20 30 40\n"
            "Epochs\n"
            "baseline\n"
            "ours\n\n"
            "Real prose after the plot."
        )
        dropped = find_figure_lines(text)
        assert {"Accuracy", "Epochs", "baseline", "ours"} <= set(dropped)
        out = normalize(text)
        assert "Real prose introducing" in out and "Real prose after" in out

    def test_prose_beside_ticks_is_not_a_label(self):
        text = "0.2 0.4 0.6 0.8\nThe measurements were taken over a single week."
        assert find_figure_lines(text) == ["0.2 0.4 0.6 0.8"]

    def test_a_heading_after_a_caption_survives(self):
        """Captions never seed the sweep, precisely so this heading is safe."""
        text = "Figure 5: Throughput.\nDiscussion\nThe results were encouraging."
        dropped = find_figure_lines(text)
        assert "Discussion" not in dropped
        assert "Discussion" in normalize(text)

    def test_a_blank_line_stops_the_sweep(self):
        text = "0.2 0.4 0.6 0.8\n\nDiscussion\n\nThe results were encouraging."
        assert "Discussion" not in find_figure_lines(text)

    def test_the_sweep_is_bounded(self):
        """A long column of short lines must not be eaten wholesale by one tick line."""
        items = "\n".join(f"Item {w}" for w in
                          ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"))
        # No digits in the item names, so only the cap stops the cascade.
        text = "0.2 0.4 0.6 0.8\n" + items.replace("Item ", "")
        dropped = find_figure_lines(text)
        assert len(dropped) <= 1 + 6  # the tick line plus at most _MAX_LABELS_PER_SIDE

    def test_punctuated_lines_are_not_labels(self):
        text = "0.2 0.4 0.6 0.8\nThe end.\nAnd more."
        assert find_figure_lines(text) == ["0.2 0.4 0.6 0.8"]


class TestFigureFalsePositives:
    """Cases found by auditing a real EPUB run log. Each one dropped live prose."""

    def test_inline_markup_does_not_shatter_a_sentence(self):
        """EPUB chapters wrap words in <i>/<sup>; those must not become line breaks."""
        from narratortool.extract.html import html_to_text

        out = html_to_text(
            "<p>He endeav<i>our</i>ed to arouse himself, "
            "known as <i>cybernetics</i>, or <i>connectionism</i>.</p>"
        )
        assert "endeavoured" in out
        assert "cybernetics, or connectionism." in out
        assert "\n" not in out.strip()

    def test_a_stray_punctuation_line_does_not_eat_its_neighbours(self):
        """A lone "," from an extractor seeded the sweep and ate the prose around it."""
        text = "always\n,\nwords\ndeep learning\ncybernetics"
        assert find_figure_lines(text) == [","]

    def test_a_heading_starting_with_a_figure_word_is_not_a_caption(self):
        for heading in ("ALGORITHMIC CREATIVITY?", "TABLES AND CHAIRS", "Charting A Course"):
            assert not find_figure_lines(heading), heading

    def test_a_word_containing_a_digit_is_not_a_number(self):
        for line in ("—@DougBlank2", "COVID19 changed everything", "R2D2 beeped"):
            assert not find_figure_lines(line), line

    def test_prose_naming_a_hyphenated_figure_is_kept(self):
        line = "Figure P2-1 highlights six roles found in the missing middle."
        assert not find_figure_lines(line)

    def test_hyphenated_captions_still_go(self):
        assert find_figure_lines("Figure P2-1: The missing middle")
        assert find_figure_lines("Figure 4 - Results")

    def test_index_entries_are_still_suppressed(self):
        text = "Accenture, 44, 47, 178, 212-213, 236\nadaptive processes, 8-10, 43-44, 51"
        assert len(find_figure_lines(text)) == 2
