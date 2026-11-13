"""Synthetic feature schema for the CTAID POC demo model.

These columns were invented for this portfolio demonstration. They are not
copied from any private research dataset and do not describe real people.
"""

from typing import Any

FEATURE_SPECS: list[dict[str, Any]] = [
    {
        "name": "age_years",
        "label": "Age (years)",
        "kind": "number",
        "min": 30,
        "max": 90,
        "step": 1,
        "default": 58,
    },
    {
        "name": "systolic_bp",
        "label": "Systolic blood pressure (mmHg)",
        "kind": "number",
        "min": 90,
        "max": 200,
        "step": 1,
        "default": 132,
    },
    {
        "name": "total_cholesterol",
        "label": "Total cholesterol (mg/dL)",
        "kind": "number",
        "min": 120,
        "max": 350,
        "step": 1,
        "default": 210,
    },
    {
        "name": "calcium_score",
        "label": "Synthetic calcium score",
        "kind": "number",
        "min": 0,
        "max": 800,
        "step": 1,
        "default": 80,
        "hint": "A made-up numeric score used only in this demo.",
    },
    {
        "name": "stenosis_pct",
        "label": "Synthetic vessel narrowing (%)",
        "kind": "number",
        "min": 0,
        "max": 90,
        "step": 1,
        "default": 20,
        "hint": "A made-up percentage used only in this demo.",
    },
    {
        "name": "smoking",
        "label": "Current smoking",
        "kind": "binary",
        "default": 0,
    },
    {
        "name": "diabetes",
        "label": "Diabetes",
        "kind": "binary",
        "default": 0,
    },
]

FEATURE_NAMES = [s["name"] for s in FEATURE_SPECS]

# Named synthetic profiles for the "load a sample" control. All invented.
SYNTHETIC_SAMPLES: list[dict[str, Any]] = [
    {
        "id": "lower-risk-sample",
        "title": "Lower-risk synthetic sample",
        "age_years": 42,
        "systolic_bp": 118,
        "total_cholesterol": 175,
        "calcium_score": 8,
        "stenosis_pct": 5,
        "smoking": 0,
        "diabetes": 0,
    },
    {
        "id": "moderate-sample",
        "title": "Moderate synthetic sample",
        "age_years": 61,
        "systolic_bp": 138,
        "total_cholesterol": 225,
        "calcium_score": 120,
        "stenosis_pct": 28,
        "smoking": 0,
        "diabetes": 1,
    },
    {
        "id": "higher-risk-sample",
        "title": "Higher-risk synthetic sample",
        "age_years": 74,
        "systolic_bp": 162,
        "total_cholesterol": 268,
        "calcium_score": 410,
        "stenosis_pct": 62,
        "smoking": 1,
        "diabetes": 1,
    },
]
