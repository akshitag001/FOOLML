# Project website

Static site (plain HTML, CSS and JavaScript) explaining our hard-label black-box attack on the CIFAR-10 model.

- `index.html` holds all the text. Placeholders to fill in are in `[square brackets]` and marked with `<!-- EDIT: -->` comments.
- `style.css` holds the design. Colours and fonts are at the top, in `:root`.
- `script.js` loads `data/*.json` and builds the image grids, tables, attack player and chart.
- `data/` is written by the last cell of `p2_code.ipynb` ("Export data for the project website"). Re-run that cell to refresh it.

## Test locally

Browsers block `fetch()` for files opened directly from disk, so serve the folder instead:

```
cd site
python -m http.server 8000
```

Then open http://localhost:8000.

## Deploy

The workflow `.github/workflows/pages.yml` publishes this folder to GitHub Pages on every push to `main`. In the repository, go to Settings → Pages and set Source to "GitHub Actions".
