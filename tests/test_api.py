from fastapi.testclient import TestClient

from app.features import SYNTHETIC_SAMPLES
from app.main import app

client = TestClient(app)


def test_home_has_banner_and_english_copy():
    res = client.get("/")
    assert res.status_code == 200
    body = res.text
    assert "Demo with synthetic data, not a medical tool." in body
    assert "CTAID POC" in body
    assert "UNISEL" not in body
    assert "ML_All" not in body
    assert chr(0x2014) not in body and chr(0x2013) not in body


def test_health_and_version():
    assert client.get("/health").json() == {"status": "ok"}
    assert "version" in client.get("/version.json").json()


def test_predict_lower_and_higher_samples():
    lower = {k: v for k, v in SYNTHETIC_SAMPLES[0].items() if k not in ("id", "title")}
    higher = {k: v for k, v in SYNTHETIC_SAMPLES[2].items() if k not in ("id", "title")}
    a = client.post("/api/predict", json=lower)
    b = client.post("/api/predict", json=higher)
    assert a.status_code == 200
    assert b.status_code == 200
    assert "not a medical tool" in a.json()["disclaimer"]
    assert a.json()["probability"] < b.json()["probability"]
    assert a.json()["probability"] < 0.5
    assert b.json()["probability"] >= 0.5


def test_predict_rejects_out_of_range():
    payload = {k: v for k, v in SYNTHETIC_SAMPLES[1].items() if k not in ("id", "title")}
    payload["age_years"] = 12
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
