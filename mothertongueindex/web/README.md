# MotherTongueIndex website

A light Material 3 Expressive site: landing, interactive analyzer, real results,
docs, and full SEO / AEO / GEO metadata.

## Run with the live analyzer

```bash
pip install "mothertongueindex[web]"   # or: pip install fastapi "uvicorn[standard]"
python web/server.py                   # http://localhost:8000
```

The live server tokenizes any pasted text with the real tokenizers via
`POST /api/analyze`.

## Static hosting (GitHub Pages, any static host)

The site works without a server. The Results table and the sample languages use
precomputed real data baked into `assets/tables.js`. Only "paste your own
arbitrary text" needs the live server.

Regenerate the baked data after changing samples or models:

```bash
python data/build_tables.py --models gpt-4o,gpt-4,claude,gemini
python - <<'PY'
import json
t = json.load(open('data/tables/by_language.json', encoding='utf-8'))
s = json.load(open('data/samples.json', encoding='utf-8'))['samples']
open('web/assets/tables.js','w',encoding='utf-8').write(
    'window.MTI_TABLES = %s;\nwindow.MTI_SAMPLES = %s;\n'
    % (json.dumps(t, ensure_ascii=False), json.dumps(s, ensure_ascii=False)))
PY
```

## Files

```
web/
├── index.html         semantic page + JSON-LD (SoftwareApplication, FAQPage, Breadcrumb)
├── assets/
│   ├── styles.css     Material 3 Expressive, light scheme, responsive
│   ├── app.js         interactive analyzer (live API + static fallback)
│   ├── tables.js      baked real data (generated)
│   └── logo.svg
├── server.py          FastAPI: serves site + /api/analyze
├── robots.txt         crawlers welcome, including AI engines
├── sitemap.xml
└── llms.txt           machine-readable summary for AI answer engines
```

## Design notes

- Light colour scheme only, by request. Material 3 style tokens, pill buttons,
  large radii, tonal surfaces, springy motion, `prefers-reduced-motion` honoured.
- Mobile-first and responsive; the comparison table scrolls inside its own
  container so the page never scrolls sideways.
- No external fonts or scripts: fast, private, offline-friendly.
- SEO: title, description, canonical, Open Graph, Twitter card. AEO/GEO: FAQ
  schema, breadcrumb, `llms.txt`, crawlable real content and data tables.
