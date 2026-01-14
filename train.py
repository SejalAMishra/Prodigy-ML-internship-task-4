import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'training')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'validation')
SAVE_PATH = os.path.join(BASE_DIR, 'model', 'food_model.h5')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 11

def train_model():
    # 1. Verify Dataset Exists
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"[ERROR] Dataset not found!")
        print(f"Please create a 'dataset' folder in '{BASE_DIR}'")
        print("Inside it, place 'training' and 'validation' folders from Food-11.")
        return

    print(f"[INFO] Training on GPU: {tf.config.list_physical_devices('GPU')}")
    
    # 2. Data Generators (Augmentation for Training)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print("[INFO] Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # 3. Build Model (MobileNetV2 Transfer Learning)
    print("[INFO] Building Model...")
    # 

[Image of MobileNetV2 architecture]

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model to keep pre-trained features
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x) # Dropout helps prevent overfitting
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train
    print(f"[INFO] Starting Training for {EPOCHS} epochs...")
    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Saving current progress...")

    # 5. Save Model
    model.save(SAVE_PATH)
    print(f"[SUCCESS] Model saved to: {SAVE_PATH}")
    
    # Print Class Mapping (Useful to verify your app.py list matches this)
    print("\n[NOTE] Class Indices Mapping:")
    print(train_generator.class_indices)

if __name__ == "__main__":
    train_model()