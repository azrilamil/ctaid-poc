from pathlib import Path

from app.synthetic import MODEL_PATH, generate_dataset, train_and_save


def test_generator_is_deterministic_and_labelled():
    X1, y1 = generate_dataset(n=200, seed=42)
    X2, y2 = generate_dataset(n=200, seed=42)
    assert X1.shape == (200, 7)
    assert set(y1.tolist()) <= {0, 1}
    assert (X1 == X2).all()
    assert (y1 == y2).all()
    # Both classes should appear in a 200-row draw.
    assert 0 in y1 and 1 in y1


def test_trained_model_is_better_than_chance():
    metrics = train_and_save(MODEL_PATH)
    assert metrics["source"] == "synthetic-generator-only"
    assert metrics["roc_auc"] >= 0.75
    assert Path(MODEL_PATH).is_file()
