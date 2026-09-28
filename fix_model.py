import tensorflow as tf
import os

SOURCE = "Model/kaggle_model.h5"
DESTINATION = "Model/kaggle_model_fixed.h5"


class CompatibleInputLayer(tf.keras.layers.InputLayer):

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("optional", None)

        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")

        return cls(**config)


CUSTOM_OBJECTS = {
    "InputLayer": CompatibleInputLayer,
    "CompatibleInputLayer": CompatibleInputLayer,
    "keras.layers.InputLayer": CompatibleInputLayer,
}


def fix_model():
    print("\nLoading Kaggle model...")

    if not os.path.exists(SOURCE):
        raise FileNotFoundError(SOURCE)

    model = tf.keras.models.load_model(
        SOURCE,
        compile=False,
        custom_objects=CUSTOM_OBJECTS
    )

    print("Original model loaded!")
    print("Input:", model.input_shape)
    print("Output:", model.output_shape)

    model.save(DESTINATION, save_format="h5")

    print("\nModel saved:", DESTINATION)

    fixed_model = tf.keras.models.load_model(
        DESTINATION,
        compile=False,
        custom_objects=CUSTOM_OBJECTS
    )

    print("\nKAGGLE MODEL FIXED AND VERIFIED SUCCESSFULLY!")
    print("Input shape:", fixed_model.input_shape)
    print("Output shape:", fixed_model.output_shape)


if __name__ == "__main__":
    fix_model()