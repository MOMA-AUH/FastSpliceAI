import unittest

import numpy as np
import torch

import spliceai.model as model_module

torch.set_num_threads(2)


class TestEnsembleModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model_module.EnsembleSpliceAIModel()

    def test_loads_all_five_frozen_members(self):
        self.assertEqual(len(self.model.members), 5)
        self.assertFalse(self.model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.model.parameters())
        )

    def test_predictions_match_keras_reference(self):
        expected_members = np.asarray(
            [
                [1.0, 2.929875675405924e-09, 9.004444967430913e-10],
                [0.9999996423721313, 3.5302875289744406e-07, 1.510788116831918e-08],
                [1.0, 5.675126590887203e-09, 1.4176610108052046e-08],
                [1.0, 4.772240824735263e-09, 3.0840405229604073e-10],
                [1.0, 1.1821275069934245e-09, 3.7705635946849725e-08],
            ],
            dtype=np.float32,
        )
        random = np.random.default_rng(20240807)
        bases = random.integers(0, 4, size=10001)
        inputs = np.eye(4, dtype=np.float32)[bases][None]

        predictions = []
        channels_first = torch.from_numpy(inputs).transpose(1, 2)
        with torch.inference_mode():
            for member in self.model.members:
                prediction = member(channels_first).transpose(1, 2).numpy()
                predictions.append(prediction[0, 0])

        np.testing.assert_allclose(predictions, expected_members, rtol=1e-4, atol=1e-12)
        with torch.inference_mode():
            ensemble_prediction = self.model(torch.from_numpy(inputs)).numpy()
        np.testing.assert_allclose(
            ensemble_prediction[0, 0],
            expected_members.mean(axis=0),
            rtol=1e-4,
            atol=1e-12,
        )
        np.testing.assert_allclose(ensemble_prediction.sum(axis=-1), 1.0)


class TestMixedPrecisionInference(unittest.TestCase):
    def test_cpu_probabilities_match_float32(self):
        random = np.random.default_rng(20240807)
        bases = random.integers(0, 4, size=(1, 10021))
        inputs = np.eye(4, dtype=np.float32)[bases]
        inputs[0, ::7] = 0

        model = model_module.EnsembleSpliceAIModel().to("cpu")
        tensor = torch.from_numpy(inputs)
        with torch.inference_mode():
            float32_predictions = model(tensor).numpy()
        with torch.inference_mode(), torch.autocast(device_type="cpu"):
            mixed_precision_predictions = model(tensor).to(torch.float32).numpy()

        self.assertEqual(mixed_precision_predictions.shape, (1, 21, 3))
        self.assertTrue(np.isfinite(mixed_precision_predictions).all())
        np.testing.assert_allclose(
            mixed_precision_predictions,
            float32_predictions,
            rtol=2e-3,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
