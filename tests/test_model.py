"""Tests for the saved model artifact (models/loan_approval_model.joblib)."""

import joblib
from sklearn.pipeline import Pipeline


class TestModelArtifact:
    def test_model_file_exists(self, model_path):
        assert model_path.exists()

    def test_model_can_be_loaded_with_joblib(self, model_path):
        pipeline = joblib.load(model_path)
        assert isinstance(pipeline, Pipeline)

    def test_loaded_pipeline_has_preprocessing_and_model_steps(self, trained_pipeline):
        assert "preprocessing" in trained_pipeline.named_steps
        assert "model" in trained_pipeline.named_steps


class TestModelPrediction:
    def test_can_predict_on_a_single_applicant(self, trained_pipeline, sample_applicant):
        prediction = trained_pipeline.predict(sample_applicant)
        assert len(prediction) == 1

    def test_prediction_is_binary(self, trained_pipeline, sample_applicant):
        prediction = trained_pipeline.predict(sample_applicant)
        assert prediction[0] in (0, 1)

    def test_predict_proba_returns_a_valid_probability(self, trained_pipeline, sample_applicant):
        probability = trained_pipeline.predict_proba(sample_applicant)[0][1]
        assert 0.0 <= probability <= 1.0
