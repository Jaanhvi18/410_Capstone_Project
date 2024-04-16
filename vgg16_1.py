import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

church_test = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/test/church/images"
lizard_test = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/test/lizard/images"
church_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/church/images"
pond_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/pond/images"
lizard_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/lizard/images"
dioscuri_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/dioscuri/images"
cup_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/transp_obj_glass_cup/images"
cylinder_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/transp_obj_glass_cylinder/images"
temple_train = "/datalake/datasets/ML_Davis/image-matching-challenge-2024/train/multi-temporal-temple-baalshamin/images"

# New function to create a model using VGG16 for feature extraction
def create_vgg16_based_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model layers
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Modify image loading function to preprocess with VGG16 requirements
def load_images_and_labels(base_dirs, target_size=(224, 224)):
    images = []
    labels = []
    for base_dir in base_dirs:
        if os.path.isdir(base_dir):
            for file_name in sorted(os.listdir(base_dir)):
                file_path = os.path.join(base_dir, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = image.load_img(file_path, target_size=target_size)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)  # Use VGG16 preprocess
                    images.append(img_array)
                    label = 1 if 'church' in base_dir else 0
                    labels.append(label)
    images = np.vstack(images)  # Combine all image arrays
    return images, np.array(labels)

# ... [Your directory path assignments here, unchanged] ...
train_dirs = [church_train, lizard_train, pond_train, dioscuri_train, cup_train, cylinder_train, temple_train]
test_dirs = [church_test, lizard_test]
# Load training and testing images
train_images, train_labels = load_images_and_labels(train_dirs)
test_images, test_labels = load_images_and_labels(test_dirs)

# Calculate steps per epoch and validation steps
batch_size = 20
steps_per_epoch = max(len(train_images) // batch_size, 1)  # Prevent zero division
validation_steps = max(len(test_images) // batch_size, 1) 

# Ensure that you have at least 1 step even for small datasets
steps_per_epoch = max(steps_per_epoch, 1)
validation_steps = max(validation_steps, 1)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# Use VGG16-specific preprocessing
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size, shuffle=False)

# Create and compile the new VGG16-based model
model = create_vgg16_based_model((224, 224, 3))

# Fit model
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10)

# Evaluate model
results = model.evaluate(test_generator, steps=validation_steps)
print("Test loss, test acc:", results)

# Predict on the test data
predictions = model.predict(test_generator, steps=validation_steps)
print(predictions)

# Single image prediction example (use VGG16 preprocessing here as well)
# Assume 'model' is your trained VGG16-based model

img_path = '00734.png'
img = image.load_img(img_path, target_size=(224, 224))  # Update to VGG16 size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print(f"{prediction[0]} The image is classified as a Church.")
else:
    print(f"{prediction[0]} The image is not classified as a Church.")