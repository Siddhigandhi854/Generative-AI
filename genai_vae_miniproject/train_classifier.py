from pathlib import Path

import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
classifier.fit(
    x_train[..., None],
    y_train,
    epochs=2,
    batch_size=256,
    validation_split=0.1,
    verbose=2,
)

_, accuracy = classifier.evaluate(x_test[..., None], y_test, verbose=0)
print(f"MNIST test accuracy: {accuracy:.4f}")
classifier.save(Path(__file__).parent / "models" / "digit_classifier.keras")
print("Saved models/digit_classifier.keras")
