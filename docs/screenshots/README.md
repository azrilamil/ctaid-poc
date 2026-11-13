# Screenshots

Slots for captures of the running app, one per page.

The `.svg` files in this folder are generated placeholders
(`python scripts/make_screenshot_placeholders.py`). They exist so the README
table renders before real captures land, and they are clearly labelled as
placeholders.

Replace each slot with a real PNG capture from the deployed demo, keeping the
same base name so the README links stay valid.

| File | Page | What the capture should show |
| --- | --- | --- |
| `home.png` | Home | Hero band, four stat cards, class balance chart |
| `data-exploration.png` | Data Exploration | Distribution tab with an overlaid histogram and a box plot |
| `model-training.png` | Model Training | Metric comparison chart plus one model's confusion matrix |
| `prediction.png` | Prediction | Class probability bars and the predicted pattern card |

Rules for captures:

1. Take the screenshot from the deployed app at https://ctaid-poc.azril.io or
   from a local run, never from a design mock up.
2. Keep the demo notice visible in every capture.
3. Do not include browser chrome, bookmarks, tabs, or any personal information.
4. Full width, at least 1440 pixels, PNG.
5. When a PNG lands, delete the matching placeholder SVG and update the README
   table to point at the PNG.
