from importlib.resources import files

import tensorflow as tf
from keras import Input, Model
from keras.layers import Average
from keras.models import load_model

from spliceai import name


class EnsembleModel(Model):
    def __init__(self, **kwargs):
        members = []
        for idx in range(1, 6):
            member = load_model(
                files(name).joinpath(f"models/spliceai{idx}.h5"),
                compile=False,
            )
            member.name = f"spliceai_model_{idx}"
            member.trainable = False
            members.append(member)

        ensemble_input = Input(
            shape=members[0].input_shape[1:],
            dtype=members[0].inputs[0].dtype,
            name="ensemble_input",
        )

        predictions = [member(ensemble_input, training=False)[0] for member in members]
        ensemble_output = Average(name="ensemble_output")(predictions)

        kwargs.setdefault("name", "spliceai_ensemble")
        super().__init__(
            inputs=ensemble_input,
            outputs=ensemble_output,
            **kwargs,
        )

    @tf.function(
        input_signature=[tf.TensorSpec((None, None, 4), tf.float32)],
        reduce_retracing=True,
        jit_compile=False,
    )
    def _infer(self, inputs):
        return self(inputs, training=False)

    def infer(self, inputs):
        return self._infer(inputs).numpy()
