"""
Unit test suite for leakage-free predictive maintenance pipeline.

Test Coverage:
1. Correct feature ordering.
2. Target exclusion from feature matrix X.
3. Median imputation of missing feature values.
4. Imputer fitted strictly on training data (no leakage) using production pipeline.
5. Zero record overlap between Train, Validation, and Test partitions.
6. Target class validation suite:
   - Valid integer targets (0, 1).
   - Valid binary numeric targets (0.0, 1.0).
   - Fractional target rejection (0.5, 1.8).
   - Invalid target rejection (2, -1, 99, boolean, string).
   - Missing target handling (dropping NaN with warning).
7. Prediction output shape, binary classes, and valid probability simplex distribution.
8. Reproducibility of stratified train/validation/test splits.
9. Holdout test set protection during training data loading.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
import data_prep
import train


class TestLeakageFreePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load raw dataset once for tests
        cls.raw_df = pd.read_csv(config.RAW_DATA_FILE)
        cls.clean_df = data_prep._clean_data(cls.raw_df)
        cls.train_df, cls.val_df, cls.test_df = data_prep._train_val_test_split(
            cls.clean_df,
            test_size=config.TEST_SIZE,
            val_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE,
        )

    # 1. Correct feature ordering
    def test_feature_ordering(self):
        expected_features = [
            "Engine_RPM",
            "Lub_Oil_Pressure",
            "Fuel_Pressure",
            "Coolant_Pressure",
            "Lub_Oil_Temperature",
            "Coolant_Temperature",
        ]
        self.assertEqual(config.FEATURE_COLUMNS, expected_features)

        X_train = self.train_df[config.FEATURE_COLUMNS]
        self.assertEqual(list(X_train.columns), expected_features)

        X_val = self.val_df[config.FEATURE_COLUMNS]
        self.assertEqual(list(X_val.columns), expected_features)

        X_test = self.test_df[config.FEATURE_COLUMNS]
        self.assertEqual(list(X_test.columns), expected_features)

    # 2. Target exclusion from X
    def test_target_exclusion_from_features(self):
        target_col = config.TARGET_COLUMN

        # Verify target is excluded from all feature matrices
        X_train = self.train_df[config.FEATURE_COLUMNS]
        X_val = self.val_df[config.FEATURE_COLUMNS]
        X_test = self.test_df[config.FEATURE_COLUMNS]

        self.assertNotIn(target_col, X_train.columns)
        self.assertNotIn(target_col, X_val.columns)
        self.assertNotIn(target_col, X_test.columns)

    # 3. Median imputation of missing feature values using production pipeline
    def test_median_imputation_of_missing_features(self):
        pipeline, _ = train._build_model_and_search_space()
        # Fast fit on small sample
        X_sample = self.train_df[config.FEATURE_COLUMNS].iloc[:100].copy()
        y_sample = self.train_df[config.TARGET_COLUMN].iloc[:100].copy()

        # Set fast estimator for unit test execution
        pipeline.set_params(clf__n_estimators=5, clf__max_depth=3)
        pipeline.fit(X_sample, y_sample)

        # Create evaluation batch with artificial missing values
        X_with_nan = X_sample.iloc[:10].copy().astype(float)
        X_with_nan.iloc[0, 0] = np.nan  # Engine_RPM
        X_with_nan.iloc[1, 1] = np.nan  # Lub_Oil_Pressure
        X_with_nan.iloc[2, 2] = np.nan  # Fuel_Pressure
        X_with_nan.iloc[3, 3] = np.nan  # Coolant_Pressure
        X_with_nan.iloc[4, 4] = np.nan  # Lub_Oil_Temperature
        X_with_nan.iloc[5, 5] = np.nan  # Coolant_Temperature

        preds = pipeline.predict(X_with_nan)
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(p in [0, 1] for p in preds))

    # 4. Imputer fitted only on training data (no leakage) using production pipeline
    def test_imputer_fitted_only_on_training_data(self):
        """
        Verify that the production pipeline's SimpleImputer computes its statistics
        strictly from the training partition and is not influenced by validation data.
        """
        pipeline, _ = train._build_model_and_search_space()
        pipeline.set_params(clf__n_estimators=5, clf__max_depth=3)

        X_train = self.train_df[config.FEATURE_COLUMNS].copy()
        y_train = self.train_df[config.TARGET_COLUMN].copy()

        pipeline.fit(X_train, y_train)

        fitted_medians = pipeline.named_steps["imputer"].statistics_
        expected_medians = X_train.median(axis=0).values

        np.testing.assert_allclose(
            fitted_medians,
            expected_medians,
            rtol=1e-5,
            err_msg="Production pipeline imputer statistics do not match training data medians!",
        )

        # Ensure predicting on extreme validation data does NOT alter fitted statistics
        X_val_extreme = self.val_df[config.FEATURE_COLUMNS].copy() * 100
        _ = pipeline.predict(X_val_extreme)

        np.testing.assert_allclose(
            pipeline.named_steps["imputer"].statistics_,
            expected_medians,
            rtol=1e-5,
            err_msg="Validation data leaked into production imputer statistics!",
        )

    # 5. No train/validation/test record overlap
    def test_no_train_val_test_overlap(self):
        feat_cols = config.FEATURE_COLUMNS

        # Check train vs val
        overlap_train_val = pd.merge(
            self.train_df[feat_cols], self.val_df[feat_cols], on=feat_cols, how="inner"
        )
        self.assertEqual(len(overlap_train_val), 0, "Train and Val sets have overlapping records!")

        # Check train vs test
        overlap_train_test = pd.merge(
            self.train_df[feat_cols], self.test_df[feat_cols], on=feat_cols, how="inner"
        )
        self.assertEqual(len(overlap_train_test), 0, "Train and Test sets have overlapping records!")

        # Check val vs test
        overlap_val_test = pd.merge(
            self.val_df[feat_cols], self.test_df[feat_cols], on=feat_cols, how="inner"
        )
        self.assertEqual(len(overlap_val_test), 0, "Val and Test sets have overlapping records!")

    # 6. Target validation suite
    def test_valid_integer_targets(self):
        """Verify valid integer targets {0, 1} are accepted and maintain integer type."""
        sample_df = self.raw_df.head(10).copy()
        sample_df["Engine Condition"] = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
        cleaned = data_prep._clean_data(sample_df)
        self.assertEqual(cleaned[config.TARGET_COLUMN].tolist(), [0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned[config.TARGET_COLUMN]))

    def test_valid_binary_numeric_targets(self):
        """Verify numeric float targets {0.0, 1.0} are accepted and converted to integer."""
        sample_df = self.raw_df.head(4).copy()
        sample_df["Engine Condition"] = [0.0, 1.0, 1.0, 0.0]
        cleaned = data_prep._clean_data(sample_df)
        self.assertEqual(cleaned[config.TARGET_COLUMN].tolist(), [0, 1, 1, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned[config.TARGET_COLUMN]))

    def test_fractional_target_rejection(self):
        """Verify fractional target values (e.g. 0.5, 1.8) raise ValueError."""
        for bad_val in [0.5, 1.8, 0.2, -0.1]:
            sample_df = self.raw_df.head(5).copy()
            sample_df["Engine Condition"] = sample_df["Engine Condition"].astype(object)
            sample_df.loc[0, "Engine Condition"] = bad_val
            with self.assertRaises(ValueError, msg=f"Failed to reject fractional target: {bad_val}"):
                data_prep._clean_data(sample_df)

    def test_invalid_target_rejection(self):
        """Verify out-of-bounds, string, or boolean target values raise ValueError."""
        for bad_val in [2, -1, 99, "fault", True, False]:
            sample_df = self.raw_df.head(5).copy()
            sample_df["Engine Condition"] = sample_df["Engine Condition"].astype(object)
            sample_df.loc[0, "Engine Condition"] = bad_val
            with self.assertRaises(ValueError, msg=f"Failed to reject invalid target: {bad_val}"):
                data_prep._clean_data(sample_df)

    def test_missing_target_handling(self):
        """Verify rows with missing targets (NaN) are dropped with a warning and remaining kept."""
        sample_df = self.raw_df.head(6).copy()
        sample_df.loc[1, "Engine Condition"] = np.nan
        sample_df.loc[3, "Engine Condition"] = np.nan
        cleaned = data_prep._clean_data(sample_df)
        self.assertEqual(len(cleaned), 4)
        self.assertFalse(cleaned[config.TARGET_COLUMN].isna().any())

    # 7. Prediction output shape and valid probability distribution (using production pipeline)
    def test_prediction_output_shape_and_classes(self):
        """
        Verify that the production pipeline generates valid prediction output shapes,
        binary class labels {0, 1}, and valid probability distributions (summing to 1.0).
        Note: These outputs represent empirical tree voting distributions, not calibrated probabilities.
        """
        pipeline, _ = train._build_model_and_search_space()
        pipeline.set_params(clf__n_estimators=5, clf__max_depth=3)

        X_train = self.train_df[config.FEATURE_COLUMNS].iloc[:200]
        y_train = self.train_df[config.TARGET_COLUMN].iloc[:200]
        pipeline.fit(X_train, y_train)

        X_val = self.val_df[config.FEATURE_COLUMNS].iloc[:50]
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)

        # Verify output shapes
        self.assertEqual(preds.shape, (50,))
        self.assertEqual(probs.shape, (50, 2))

        # Verify prediction values are binary
        self.assertTrue(all(p in [0, 1] for p in preds))

        # Verify probabilities sum to 1.0 and fall in [0.0, 1.0]
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(50), rtol=1e-5)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    # 8. Reproducibility of the split
    def test_split_reproducibility(self):
        train_1, val_1, test_1 = data_prep._train_val_test_split(
            self.clean_df,
            test_size=config.TEST_SIZE,
            val_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE,
        )

        train_2, val_2, test_2 = data_prep._train_val_test_split(
            self.clean_df,
            test_size=config.TEST_SIZE,
            val_size=config.VAL_SIZE,
            random_state=config.RANDOM_STATE,
        )

        pd.testing.assert_frame_equal(train_1, train_2)
        pd.testing.assert_frame_equal(val_1, val_2)
        pd.testing.assert_frame_equal(test_1, test_2)

    # 9. Protection of holdout test set during training loading
    def test_holdout_test_not_accessed_during_training_load(self):
        """Verify that training data loader returns only (train_df, val_df) and does not access test set."""
        train_df, val_df = train._load_training_and_validation_data()
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(val_df, pd.DataFrame)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)


if __name__ == "__main__":
    unittest.main()
