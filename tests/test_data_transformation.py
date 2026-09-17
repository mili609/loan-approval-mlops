"""Tests for src/transformation/data_transformation.py."""

import numpy as np
import pandas as pd
import pytest

from src.transformation.data_transformation import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    build_preprocessing_pipeline,
    encode_target,
    split_features_and_target,
    train_test_split_data,
)


class TestEncodeTarget:
    def test_maps_y_and_n_to_1_and_0(self):
        result = encode_target(pd.Series(["y", "n", "y", "n"]))
        assert result.tolist() == [1, 0, 1, 0]

    def test_is_case_and_whitespace_insensitive(self):
        result = encode_target(pd.Series([" Y ", "N", " y", "n "]))
        assert result.tolist() == [1, 0, 1, 0]

    def test_raises_on_unexpected_value(self):
        with pytest.raises(ValueError, match="Unexpected loan_status values"):
            encode_target(pd.Series(["y", "maybe", "n"]))

    def test_returns_integer_dtype(self):
        result = encode_target(pd.Series(["y", "n"]))
        assert np.issubdtype(result.dtype, np.integer)


class TestSplitFeaturesAndTarget:
    def test_drops_identifier_and_target_columns_from_features(self, valid_loan_dataframe):
        X, y = split_features_and_target(valid_loan_dataframe)
        assert "loan_id" not in X.columns
        assert "loan_status" not in X.columns

    def test_target_is_binary_encoded(self, valid_loan_dataframe):
        _, y = split_features_and_target(valid_loan_dataframe)
        assert set(y.unique()) <= {0, 1}

    def test_row_counts_match_input(self, valid_loan_dataframe):
        X, y = split_features_and_target(valid_loan_dataframe)
        assert len(X) == len(valid_loan_dataframe)
        assert len(y) == len(valid_loan_dataframe)


class TestPreprocessingPipeline:
    @staticmethod
    def _to_dense(matrix):
        return matrix.toarray() if hasattr(matrix, "toarray") else matrix

    def test_handles_missing_values_without_raising(self, valid_loan_dataframe):
        assert valid_loan_dataframe.isnull().values.any()  # sanity check on the fixture

        X, _ = split_features_and_target(valid_loan_dataframe)
        pipeline = build_preprocessing_pipeline()
        transformed = self._to_dense(pipeline.fit_transform(X))

        assert not np.isnan(transformed).any()

    def test_categorical_features_are_one_hot_encoded(self, valid_loan_dataframe):
        X, _ = split_features_and_target(valid_loan_dataframe)
        pipeline = build_preprocessing_pipeline()
        transformed = self._to_dense(pipeline.fit_transform(X))

        # One-hot encoding expands each categorical column into multiple
        # columns, so the output must have more columns than the raw
        # numerical + categorical feature count.
        assert transformed.shape[1] > len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS)

    def test_output_row_count_matches_input(self, valid_loan_dataframe):
        X, _ = split_features_and_target(valid_loan_dataframe)
        pipeline = build_preprocessing_pipeline()
        transformed = pipeline.fit_transform(X)
        assert transformed.shape[0] == len(X)


class TestTrainTestSplitData:
    def test_split_is_stratified_within_tolerance(self, raw_dataset):
        X, y = split_features_and_target(raw_dataset)
        _, _, y_train, y_test = train_test_split_data(X, y, random_state=42)

        overall_rate = y.mean()
        assert abs(y_train.mean() - overall_rate) < 0.05
        assert abs(y_test.mean() - overall_rate) < 0.05

    def test_split_is_reproducible_with_random_state_42(self, raw_dataset):
        X, y = split_features_and_target(raw_dataset)

        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split_data(
            X, y, random_state=42
        )
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split_data(
            X, y, random_state=42
        )

        assert X_train_1.index.tolist() == X_train_2.index.tolist()
        assert X_test_1.index.tolist() == X_test_2.index.tolist()
        assert y_test_1.tolist() == y_test_2.tolist()

    def test_split_sizes_match_test_size_ratio(self, raw_dataset):
        X, y = split_features_and_target(raw_dataset)
        X_train, X_test, _, _ = train_test_split_data(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)
        assert len(X_test) == pytest.approx(len(X) * 0.2, abs=1)
