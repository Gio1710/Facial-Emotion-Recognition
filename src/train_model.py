import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
BASE_DIR = '/kaggle/input/human-face-emotions' # Change this to your local path if needed
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 64
COLOR_MODE = 'rgb'
EPOCHS_HEAD = 30
EPOCHS_FINE_TUNE = 15

def get_data_generators(base_dir):
    """Creates and optimizes training and validation datasets."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        label_mode='categorical'
    )

    # Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def calculate_class_weights(train_ds):
    """Calculates class weights to handle dataset imbalance."""
    train_labels = []
    # Note: Iterating over the dataset to extract labels might take time
    for _, labels in train_ds:
        train_labels.extend(np.argmax(labels.numpy(), axis=1))
    
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    return dict(enumerate(class_weights_array))

def build_model(input_shape, num_classes):
    """Builds the ResNet50V2 model with custom classification head."""
    base_model = ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    model = Sequential([
        Rescaling(1./127.5, offset=-1, input_shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history, title="Model Performance"):
    """Plots accuracy and loss metrics."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.show()

def main():
    # 1. Data Preparation
    train_ds, val_ds = get_data_generators(BASE_DIR)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    
    print(f"Classes: {class_names}")
    
    # 2. Calculate Weights
    print("Calculating class weights...")
    class_weights = calculate_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    # 3. Build Model
    model = build_model(input_shape, num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # 4. Phase 1 Training (Head)
    print("Starting Phase 1: Training Head...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    # 5. Phase 2 Training (Fine-Tuning)
    print("Starting Phase 2: Fine-Tuning...")
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze earlier layers
    fine_tune_at = 140
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=Adam(learning_rate=1e-5), # Low LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    fine_tune_early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    initial_epoch = history_head.epoch[-1] + 1
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch + EPOCHS_FINE_TUNE,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=[fine_tune_early_stopping]
    )

    # 6. Save Model
    model.save('emotion_recognition_model_v13.keras')
    print("Model saved successfully.")

    # 7. Evaluation
    print("Evaluating on Validation Set...")
    y_true = []
    for images, labels in val_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()
