import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory where the images are located
base_dir = "/Users/kaijie/Desktop/Colgate/cosc410/project/data"


def create_binary_classification_cnn(input_shape):
    model = models.Sequential()

    # Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer with 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer with 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    # Flattening the 3D output to 1D and adding a dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))

    # Output layer with 1 unit and sigmoid activation function for binary classification
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


# Example input shape of an image (e.g., 128x128 pixels with 3 channels for RGB)
input_shape = (128, 128, 3)
model = create_binary_classification_cnn(input_shape)

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Summary of the model
model.summary()


# Image preprocessing and augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize the images
    rotation_range=40,  # Data augmentation: rotation
    width_shift_range=0.2,  # Data augmentation: horizontal shift
    height_shift_range=0.2,  # Data augmentation: vertical shift
    shear_range=0.2,  # Data augmentation: shearing
    zoom_range=0.2,  # Data augmentation: zooming
    horizontal_flip=True,  # Data augmentation: horizontal flip
    fill_mode="nearest",  # Fill in missing pixels after a transformation
)

# Assuming the input shape of the model is 128x128 pixels
target_size = (128, 128)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=target_size,  # Resize images to 128x128
    batch_size=20,
    class_mode="binary",  # Binary classification (churches vs lizards)
)

# Now you can use train_generator to fit your model
# Example:
model.fit(
    train_generator,
    steps_per_epoch=100,  # Number of steps per epoch, depend on your data size and batch size
    epochs=10,  # Number of epochs to train the model
)
