"""Tests for src/validation/data_validation.py."""

import pandas as pd
import pytest

from src.validation.data_validation import (
    REQUIRED_COLUMNS,
    validate_columns,
    validate_dataset,
    validate_duplicates,
    validate_target,
)


class TestValidateColumns:
    def test_passes_when_all_required_columns_present(self, valid_loan_dataframe):
        assert validate_columns(valid_loan_dataframe) is True

    def test_raises_when_a_required_column_is_missing(self, valid_loan_dataframe):
        df = valid_loan_dataframe.drop(columns=["credit_history"])
        with pytest.raises(ValueError, match="credit_history"):
            validate_columns(df)

    def test_error_message_lists_all_missing_columns(self, valid_loan_dataframe):
        df = valid_loan_dataframe.drop(columns=["gender", "married"])
        with pytest.raises(ValueError) as exc_info:
            validate_columns(df)
        assert "gender" in str(exc_info.value)
        assert "married" in str(exc_info.value)


class TestValidateTarget:
    def test_passes_with_fully_populated_target(self, valid_loan_dataframe):
        assert validate_target(valid_loan_dataframe) is True

    def test_raises_when_target_column_missing(self, valid_loan_dataframe):
        df = valid_loan_dataframe.drop(columns=["loan_status"])
        with pytest.raises(ValueError, match="loan_status"):
            validate_target(df)

    def test_raises_when_target_has_missing_values(self, valid_loan_dataframe):
        df = valid_loan_dataframe.copy()
        df.loc[0, "loan_status"] = None
        with pytest.raises(ValueError, match="missing values"):
            validate_target(df)


class TestValidateDuplicates:
    def test_passes_without_duplicate_rows(self, valid_loan_dataframe):
        assert validate_duplicates(valid_loan_dataframe) is True

    def test_raises_when_duplicate_rows_present(self, valid_loan_dataframe):
        df_with_duplicate = pd.concat(
            [valid_loan_dataframe, valid_loan_dataframe.iloc[[0]]],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="duplicate"):
            validate_duplicates(df_with_duplicate)


class TestValidateDataset:
    def test_valid_dataset_passes_all_checks(self, valid_loan_dataframe):
        assert validate_dataset(valid_loan_dataframe) is True

    def test_real_raw_dataset_passes_validation(self, raw_dataset):
        assert validate_dataset(raw_dataset) is True

    def test_required_columns_matches_real_dataset_schema(self, raw_dataset):
        # Guards against REQUIRED_COLUMNS silently drifting from the real schema.
        assert set(REQUIRED_COLUMNS) == set(raw_dataset.columns)
