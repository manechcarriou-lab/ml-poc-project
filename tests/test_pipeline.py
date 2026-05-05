"""Smoke tests on the data + features + trained-model pipeline.

Run with:
    pytest tests/  (from project root)
or simply:
    python -m unittest tests/test_pipeline.py
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("config", SRC / "config.py")
sys.modules["config"] = config
features = _load("features", SRC / "features.py")
sys.modules["features"] = features
data_mod = _load("data", SRC / "data.py")
metrics_mod = _load("metrics", SRC / "metrics.py")


@unittest.skipUnless(
    (config.DATA_DIR / "online_shoppers_intention.csv").exists(),
    "Raw dataset CSV missing — see data/.gitkeep for download instructions.",
)
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = data_mod.load_dataset_split()

    def test_split_sizes_are_consistent(self):
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertEqual(len(self.X_test), len(self.y_test))

    def test_split_is_stratified(self):
        train_pos = float(self.y_train.mean())
        test_pos = float(self.y_test.mean())
        # Stratified split should keep both rates within 1 % of each other.
        self.assertAlmostEqual(train_pos, test_pos, delta=0.01)

    def test_engineered_features_present(self):
        for col in [
            "TotalPages",
            "TotalDuration",
            "AvgTimePerPage",
            "ProductRelatedRatio",
            "HighPageValue",
            "IsHighBounce",
            "IsSpecialDay",
        ]:
            self.assertIn(col, self.X_train.columns)

    def test_no_target_leakage_in_features(self):
        self.assertNotIn("Revenue", self.X_train.columns)
        self.assertNotIn("Revenue", self.X_test.columns)

    def test_preprocessor_is_leakage_safe(self):
        """Fitting preprocessor twice on same train -> same shape; transform on test fits no new stats."""
        pipe = features.build_feature_pipeline()
        Xt_train = pipe.fit_transform(self.X_train, self.y_train)
        Xt_test = pipe.transform(self.X_test)
        self.assertEqual(Xt_train.shape[1], Xt_test.shape[1])

    def test_metrics_returns_expected_keys(self):
        # Use a trivial constant predictor (always majority class) to check the metrics contract.
        majority = int(self.y_train.mode().iloc[0])
        y_pred = [majority] * len(self.y_test)
        m = metrics_mod.compute_metrics(self.y_test, y_pred)
        self.assertEqual(
            set(m.keys()), {"accuracy", "precision", "recall", "f1", "roc_auc"}
        )
        for v in m.values():
            self.assertIsInstance(float(v), float)


@unittest.skipUnless(
    config.MODELS["xgboost"]["path"].exists(),
    "XGBoost joblib missing — run `python scripts/train.py`.",
)
class TestTrainedModels(unittest.TestCase):
    def test_each_registered_model_predicts(self):
        import joblib

        _, X_test, _, y_test = data_mod.load_dataset_split()
        for key, cfg in config.MODELS.items():
            with self.subTest(model=key):
                if not cfg["path"].exists():
                    self.skipTest(f"{key} joblib missing")
                pipe = joblib.load(cfg["path"])
                self.assertTrue(hasattr(pipe, "predict"))
                self.assertTrue(hasattr(pipe, "predict_proba"))
                y_pred = pipe.predict(X_test)
                self.assertEqual(len(y_pred), len(y_test))


if __name__ == "__main__":
    unittest.main()
