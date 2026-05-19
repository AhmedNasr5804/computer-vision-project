# CIE-552 Presentation — `presentation/index.html`

Self-contained futuristic-themed reveal.js deck covering the full project:
EDA, pipeline architectures, results, controlled experiments, model
selection, fine-tuning, deployment, and the four key insights that
survive past this term project.

## View it

Just open `index.html` in any modern browser (Chrome / Edge / Firefox):

```bash
# Windows
start presentation\index.html

# macOS / Linux
xdg-open presentation/index.html
```

reveal.js is loaded from CDN, fonts (Inter, Space Grotesk, JetBrains Mono)
from Google Fonts. **First load needs internet.** After that the cached
copies render offline as long as the browser's HTTP cache hasn't been
purged. To run fully offline, vendor reveal.js locally — see the optional
section at the bottom of this file.

Keyboard:

| Key | Action |
|---|---|
| <kbd>→</kbd>, <kbd>Space</kbd>, <kbd>PgDn</kbd> | next slide |
| <kbd>←</kbd>, <kbd>PgUp</kbd> | previous slide |
| <kbd>F</kbd> | toggle full-screen |
| <kbd>S</kbd> | speaker notes (none defined; opens a stub window) |
| <kbd>Esc</kbd> | overview grid (see all slides at once) |
| <kbd>B</kbd>, <kbd>.</kbd> | black-out (pause for discussion) |

## What's in the deck (63 slides)

| # | Section | Slides |
|---|---|---|
| 00 | Title + overview + talk map | 3 |
| 01 | EDA — datasets, splits, sanity grid, class balance, privacy filter | 8 |
| 02 | Pipeline architectures (eye + lane TuSimple + lane Pi) | 11 |
| 03 | Results — confusion / ROC / robustness / misclass galleries | 6 |
| 04 | Controlled experiments (A / B / C / D + sanity audit) | 9 |
| 05 | Model selection + step-by-step fine-tuning | 6 |
| 06 | Deployment — PTQ-int8, S24 benchmarks, Android app, Firebase, RPi, PIC bridge | 9 |
| 07 | Key insights — small-N / domain gap / mask regen / depth | 5 |
| FIN | Conclusion + next steps | 3 |

## Privacy

Slide 12 (S24-Ultra fine-tune set) and every other slide that touches
the eye fine-tune data renders only the **32 author-approved captures**
(prefix `20260517_*`). The original 12 captures from the prior session
(prefix `20260515_*`, family members) are used for **training only** —
they never appear in the deck or in the published paper figures.

## Export to PDF

In Chrome, append `?print-pdf` to the URL and use **Print → Save as PDF**
(set background graphics ON, margin to None, scale 100%):

```
file:///path/to/presentation/index.html?print-pdf
```

The resulting PDF preserves the futuristic theme (dark background, neon
accents). One slide per page.

## Theme reference

The deck overrides reveal.js's black theme with a custom CSS block in
`<head>`. Key tokens:

| Token | Value | Used for |
|---|---|---|
| `--bg-deep` | `#040714` | Page background |
| `--cyan` | `#00e5ff` | Primary accent (headings, metrics, borders) |
| `--magenta` | `#ff2bd6` | Secondary accent (Pipeline B, warnings, section numbers) |
| `--green` | `#4cff7f` | Positive callouts (winner labels, success metrics) |
| `--amber` | `#ffd166` | Caution callouts (failure modes, broken supervision) |

To re-theme, edit the `:root` block at the top of `index.html`.

## Optional: fully offline reveal.js

If you need to present without internet:

```bash
cd presentation
mkdir -p reveal && cd reveal
curl -L https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css       -o reveal.css
curl -L https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.css  -o black.css
curl -L https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js        -o reveal.js
```

Then in `index.html`, swap the three CDN links for `reveal/reveal.css`,
`reveal/black.css`, `reveal/reveal.js`. The Google Fonts can be vendored
the same way if needed.
