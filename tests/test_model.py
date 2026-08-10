import tempfile
import unittest
from importlib.resources import files

import h5py
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

    def test_maps_keras_convolution_and_batch_norm_weights(self):
        model_path = files("spliceai").joinpath("models/spliceai1.h5")
        with h5py.File(model_path, "r") as weights:
            expected_kernel = np.asarray(
                weights["model_weights/conv1d_1/conv1d_1/kernel:0"]
            ).transpose(2, 1, 0)
            expected_mean = np.asarray(
                weights[
                    "model_weights/batch_normalization_1/"
                    "batch_normalization_1/moving_mean:0"
                ]
            )

        np.testing.assert_array_equal(
            self.model.members[0].stem[0].module.weight.numpy(),
            expected_kernel,
        )
        np.testing.assert_array_equal(
            self.model.members[0].stem[1].module[0][0].running_mean.numpy(),
            expected_mean,
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

        np.testing.assert_allclose(
            predictions,
            expected_members,
            rtol=1e-4,
            atol=1e-12,
        )
        ensemble_prediction = self.model.infer(inputs)
        self.assertEqual(ensemble_prediction.shape, (1, 1, 3))
        np.testing.assert_allclose(
            ensemble_prediction[0, 0],
            expected_members.mean(axis=0),
            rtol=1e-4,
            atol=1e-12,
        )
        np.testing.assert_allclose(ensemble_prediction.sum(axis=-1), 1.0)

    def test_rejects_invalid_tensor_shapes(self):
        invalid_inputs = (
            torch.zeros((10001, 4)),
            torch.zeros((1, 10001, 5)),
            torch.zeros((1, 10000, 4)),
        )
        for inputs in invalid_inputs:
            with self.subTest(shape=tuple(inputs.shape)), self.assertRaises(ValueError):
                self.model(inputs)
        with self.assertRaises(TypeError):
            self.model(np.zeros((1, 10001, 4), dtype=np.float32))

    def test_rejects_empty_ensemble(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            model_module.EnsembleSpliceAIModel(model_paths=[])

    def test_reports_missing_and_malformed_keras_weights(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as weight_file:
            with h5py.File(weight_file.name, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "missing model_weights/conv1d_1"):
                model_module.EnsembleSpliceAIModel(model_paths=[weight_file.name])

        with tempfile.NamedTemporaryFile(suffix=".h5") as weight_file:
            with h5py.File(weight_file.name, "w") as weights:
                weights.create_dataset(
                    "model_weights/conv1d_1/conv1d_1/kernel:0",
                    shape=(2, 4, 32),
                    dtype="f4",
                )
            with self.assertRaisesRegex(ValueError, "has shape"):
                model_module.EnsembleSpliceAIModel(model_paths=[weight_file.name])


if __name__ == "__main__":
    unittest.main()
