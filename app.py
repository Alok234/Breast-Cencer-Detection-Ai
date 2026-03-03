import os
import zipfile
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2, EfficientNetB0, VGG16
from sklearn.model_selection import train_test_split

# === Step 1: Unzip dataset ===
zip_path = "dataset.zip"  # your zip file
extract_dir = "dataset_unzip"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset unzipped.")
else:
    print("Dataset already unzipped.")

# === Step 2: Load CSV ===
csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
if len(csv_files) == 0:
    raise Exception("No CSV file found in the zip!")
csv_path = os.path.join(extract_dir, csv_files[0])
df = pd.read_csv(csv_path)

# Map labels to 0/1
df['label'] = df['label'].map({"Normal": 0, "Abnormal": 1})

# === Step 3: Setup image directory ===
# Assumes images are somewhere inside the zip folder
image_dir = os.path.join(extract_dir, "images")  # adjust if different
if not os.path.exists(image_dir):
    # If images are in root folder of zip
    image_dir = extract_dir

# === Step 4: Prepare tf.data.Dataset ===
def load_image(filename, label):
    img_path = os.path.join(image_dir, filename)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_ds = tf.data.Dataset.from_tensor_slices((train_df['filename'].values, train_df['label'].values))
train_ds = train_ds.map(load_image).batch(16).shuffle(100)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['filename'].values, val_df['label'].values))
val_ds = val_ds.map(load_image).batch(16)

# === Step 5: Build 5-model ensemble ===
input_layer = layers.Input(shape=(224,224,3))

resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer, pooling='avg')
inception = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer, pooling='avg')
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_layer, pooling='avg')
efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_layer, pooling='avg')
vgg = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer, pooling='avg')

combined = layers.Concatenate()([resnet.output, inception.output, mobilenet.output, efficientnet.output, vgg.output])

x = layers.Dense(512, activation='relu')(combined)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Step 6: Train ensemble on dataset ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5  # increase if you want better accuracy
)

# === Step 7: Save weights ===
model.save_weights("breast_ensemble_model.h5")
print("Training complete. Weights saved as breast_ensemble_model.h5")
