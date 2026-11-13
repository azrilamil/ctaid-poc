# Data generation

`data/synthetic_features.csv` is produced by `scripts/generate_synthetic_data.py`.
Every value is invented. The script never reads an external dataset, a scan, or
a person: it samples only from the ranges stored in `app/schema.py`.

## Reproduce the committed file

```bash
python scripts/generate_synthetic_data.py
```

Defaults: `--records 700` (3 vessels per record, so 2100 rows), `--seed 20261113`,
output `data/synthetic_features.csv`.

```bash
python scripts/generate_synthetic_data.py --records 900 --out /tmp/big.csv
```

## Shape

| Item | Value |
| --- | --- |
| Rows | 2100 (700 records x 3 vessels) |
| Columns | 68 |
| Identifier column | `Record ID`, sequential from 1 to 700 |
| Grouping column | `Vessel`, values 1 to 3 |
| Feature columns | 65 numeric columns |
| Target | `Pattern`, integer 0 to 3 (stored in the CSV under the internal column name `Diagnosis`; the UI labels it `Pattern`) |

Column names, dtypes and ranges come from a published design brief. The brief
supplied names and summary statistics only, never rows: `app/schema.py` stores
`min`, `max`, `mean`, `std` and a few percentiles per column, which the generator
turns into sampling parameters.

## Sampling model

1. Each record draws a latent class from a fixed prior: 0.30, 0.26, 0.24, 0.20.
   All three vessels of a record share it, which mirrors the original grouping.
2. A positive feature with a narrow range is drawn from a log normal around the
   recorded mean, so the long right tail that dominates texture features appears
   naturally. A signed feature is drawn from a normal around the mean.
3. A small class effect is applied: a shift in standard deviation units plus a
   scale, per class, for a handful of interpretable features (mean intensity,
   maximum, a calcium style score, ROI volume, texture contrast and homogeneity).
   This is what makes the classes learnable instead of pure noise.
4. Vessels inside one record get a small shared drift, then the value is clipped
   to the schema range and integer columns are rounded.
5. Derived pairs stay consistent: the excess kurtosis columns equal the matching
   kurtosis column minus 3.

## Determinism

The seed is fixed at `20261113` and passed to `numpy.random.default_rng`. The CSV
is written with `float_format="%.6f"`. The same script, seed and NumPy release
reproduce the committed file exactly; a different NumPy version can change the
last decimals of a handful of cells, which is why the row count, column count and
class counts are what the tests assert.

## Class labels

The four classes are `Pattern A` to `Pattern D`:

| Value | Label | Description used in the UI |
| --- | --- | --- |
| 0 | Pattern A | small dense region, low spread |
| 1 | Pattern B | large dense region, high peak intensity |
| 2 | Pattern C | diffuse low density region, wide spread |
| 3 | Pattern D | mixed region, high texture contrast |

These descriptions are invented helper text for the demo. They are not clinical
categories, not clinical labels, and the app says so wherever they appear.

## What the generator does not do

- No row copying, no resampling of a real table, no interpolation of real values.
- No identifiers of any kind beyond the sequential `Record ID`.
- No images, no DICOM, no patient metadata.
- No changes to `models/`: artifacts are trained from the CSV, not shipped.

## Tests

`tests/test_generator.py` checks the column order, the row count, every value
against its schema range, integer dtypes, determinism across two draws, the
record and vessel grouping, the four class labels, and the derived column
arithmetic. It also asserts that class means actually differ, so a regression
that flattens the class signal fails the suite.
