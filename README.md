# EvoLen research website

A standalone, responsive project page for the supplied COLM 2026 paper. Uses Times New Roman throughout and a poster-inspired navy, blue, and pale-blue palette. No build step, account, package manager, analytics, or external font service required.

## Preview

Open `index.html` directly, or run from this folder:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open http://127.0.0.1:8000. A local server enables the clipboard API in supported browsers; direct-file viewing includes a selection fallback.

## Contents

- `index.html`: paper, authors, method, evidence, results, references, resources, citation.
- `styles.css`: responsive design and print styles.
- `app.js`: published motif examples, 100K/200K results toggle, figure enlargement, citation copying.
- `assets/`: five figures, paper PDF, editable poster, favicon.

The page can be uploaded as-is to static hosting. GitHub Pages publishes this site from the `gh-pages` branch of `HN020719/EvoLen`.

Project URL: https://hn020719.github.io/EvoLen/

## Scientific sources

Text and numbers come from the supplied `2604.08698v2.pdf` (28 Aug 2026). The downstream chart reproduces all 15 task-group relative improvements from Table 1. Values are transcribed rather than recalculated from rounded MCC scores. Its bars use a fixed 35% magnitude scale; decreases are explicitly negative and red. Aggregate statistics refer only to the 100K comparison. Token examples come from appendix Figure 5 and are illustrations, not a live tokenizer.

Figures are the Times New Roman derivatives from `poster_draft/figure_assets`. The region-separation chart was redrawn from the supplied figure's values in the poster workflow. The phyloP analysis tests alignment with a construction signal and is described accordingly. The full paper contains additional controls and limitations.

Edit copy in `index.html`, and benchmark values or token examples in `app.js`. Keep the 100K aggregate statistics distinct from the 200K comparison.

## Motion

The tokenization example animates once on arrival and can be replayed. Changing motifs starts the sequence again. Benchmark bars transition between their 100K and 200K values. Motion respects the operating system reduced-motion setting, and the token sequence cancels when the tab is hidden. There are no continuous animation loops.
