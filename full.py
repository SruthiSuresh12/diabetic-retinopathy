# ============================================================
# Diabetic Retinopathy Detection - Full Dataset Workflow
# ============================================================

import os
import zipfile
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import subprocess

# ============================================================
# 1. Kaggle setup
# ============================================================
competition_name = "diabetic-retinopathy-detection"
data_directory = "./diabetic-retinopathy-detection"

# Ensure Kaggle API is available
if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    raise FileNotFoundError(
        "❌ Kaggle API credentials not found. "
        "Place kaggle.json in ~/.kaggle/ before running this notebook. "
        "See: https://www.kaggle.com/docs/api"
    )

# ============================================================
# 2. Download dataset (only if not already present)
# ============================================================
labels_csv = os.path.join(data_directory, "trainLabels.csv")

if not os.path.exists(labels_csv):
    os.makedirs(data_directory, exist_ok=True)
    print(f"⬇️ Downloading {competition_name} dataset...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition_name, "-p", data_directory],
        check=True
    )
else:
    print("✅ trainLabels.csv already exists, skipping download.")

# ============================================================
# 3. Extract dataset
# ============================================================
def extract_all_zips(directory):
    extracted = True
    while extracted:
        extracted = False
        for file in os.listdir(directory):
            if file.endswith(".zip") or ".zip." in file:  # handle split archives
                zip_path = os.path.join(directory, file)
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(directory)
                    os.remove(zip_path)
                    print(f"📦 Extracted {file}")
                    extracted = True
                except zipfile.BadZipFile:
                    pass  # skip incomplete multipart zips

extract_all_zips(data_directory)

# ============================================================
# 4. Load labels
# ============================================================
train_labels_df = pd.read_csv(labels_csv)
train_dir = os.path.join(data_directory, "train")

train_labels_df["image_path"] = train_labels_df["image"].apply(
    lambda x: os.path.join(train_dir, f"{x}.jpeg")
)

# Keep only valid paths
train_labels_df = train_labels_df[train_labels_df["image_path"].apply(os.path.exists)].reset_index(drop=True)
print(f"✅ Found {len(train_labels_df)} training images")

# ============================================================
# 5. Preprocessing pipeline
# ============================================================
def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    img_green = img[:, :, 1]

    img_clahe = tf.numpy_function(
        func=lambda x: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(x.astype(np.uint8)),
        inp=[img_green],
        Tout=tf.uint8
    )
    img_clahe.set_shape(img_green.shape)
    img_processed = tf.image.grayscale_to_rgb(tf.expand_dims(img_clahe, axis=-1))

    img_resized = tf.image.resize(img_processed, (224, 224))
    img_normalized = img_resized / 255.0
    return img_normalized, label

# ============================================================
# 6. Focal Loss for class imbalance
# ============================================================
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=5)

        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        alpha_t = y_true_one_hot * self.alpha + (1 - y_true_one_hot) * (1 - self.alpha)
        p_t = y_true_one_hot * y_pred + (1 - y_true_one_hot) * (1 - y_pred)
        focal_loss = alpha_t * tf.pow((1. - p_t), self.gamma) * cross_entropy

        return tf.reduce_sum(focal_loss, axis=-1)

# ============================================================
# 7. Build ResNet50V2 model
# ============================================================
def build_model(num_classes=5):
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

# ============================================================
# 8. Build TF datasets
# ============================================================
image_paths = train_labels_df["image_path"].tolist()
labels = train_labels_df["level"].tolist()

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.shuffle(buffer_size=len(image_paths))

train_size = int(0.8 * len(image_paths))
train_ds_raw = dataset.take(train_size)
val_ds_raw = dataset.skip(train_size)

train_dataset = train_ds_raw.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_ds_raw.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ============================================================
# 9. Compile & train
# ============================================================
model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=FocalLoss(),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")]
)

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset
)

model.save("diabetic_retinopathy_model.h5")
print("✅ Training complete and model saved")
