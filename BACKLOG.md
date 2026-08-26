# Backlog

A second view of the [Issues tab](https://github.com/CryptoJones/NarratorTool/issues).
Every item here has a matching issue and every issue has an item here; when one ships,
tick the box and move it to **Done** so neither side drifts.

## Open

### Voice and pronunciation
- [ ] Proper nouns and dialect need a render-time respelling layer ([#4](https://github.com/CryptoJones/NarratorTool/issues/4)) — Chatterbox says what the letters say, so the fix is to change the model's *input*, not to correct its output. The document keeps its real spelling.

### Render QA
- [ ] Detect chunks Chatterbox truncated, which a pace check cannot see ([#5](https://github.com/CryptoJones/NarratorTool/issues/5)) — a pace check exempts short lines as noisy, and short lines are exactly where the truncation happens. Threshold 0.20 s/word, not 0.25.

## Done

- [x] Switch the default TTS backend to Chatterbox ([#1](https://github.com/CryptoJones/NarratorTool/issues/1)) — 2.7x steadier voice identity across takes, which over a book is the number that decides whether the narrator sounds like one person.
- [x] Graph vertex names narrate as letter soup ([#2](https://github.com/CryptoJones/NarratorTool/issues/2)) — a graph has no axis to seed the chart-label sweep from, so its node labels survived as prose.
- [x] Cast voice profiles, and a dry-run estimate that is 6x low ([#3](https://github.com/CryptoJones/NarratorTool/issues/3)).

---

*Proudly Made in Nebraska. Go Big Red! 🌽 <https://xkcd.com/2347/>*
