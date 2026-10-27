
▲
🙂
---
license: gpl-2.0
---

# DCGAN to generate face images

This is an example notebook for Keras sprint prepared by Hugging Face. Keras Sprint aims to reproduce Keras examples and build interactive demos to them. The markdown parts beginning with 🤗 and the following code snippets are the parts added by the Hugging Face team to give you an example of how to host your model and build a demo.

**Original Author of the DCGAN to generate face images Example:** [fchollet](https://twitter.com/fchollet)

## Steps to Train the DCGAN

1. Create the discriminator
- It maps a 64x64 image to a binary classification score.

```py

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

```

2. Create the generator
- It mirrors the discriminator, replacing Conv2D layers with Conv2DTranspose layers

```py

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

HF Contributor: [Tarun Jain](https://twitter.com/TRJ_0751)