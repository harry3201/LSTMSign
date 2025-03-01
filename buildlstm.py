from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load preprocessed data
data_dict = pickle.load(open(r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\data_sequences.pickle', 'rb'))
data = np.array(data_dict['data'])  # Shape: (num_samples, 30, 84)
labels = np.array(data_dict['labels'])

# Create label mapping
unique_labels = sorted(set(labels))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# Save label mapping
label_translations = {
    "ball": {"english": "Ball", "hindi": "गेंद", "gujarati": "બોલ"},
    "book": {"english": "Book", "hindi": "पुस्तक", "gujarati": "પુસ્તક"},
    "hurt": {"english": "Hurt", "hindi": "चोट", "gujarati": "ઈજા"}
}
with open(r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\label_mapping.pickle', 'wb') as f:
    pickle.dump(label_translations, f)

# Convert labels to integers
labels_int = np.array([label_to_int[label] for label in labels])

# One-hot encode labels
labels_onehot = to_categorical(labels_int, num_classes=len(label_to_int))

# Data Augmentation: Add noise, scaling, and slight time shifts
def augment_data(data, noise_factor=0.01, scale_factor=0.05, shift_max=2):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor, size=(data.shape[0], 1, data.shape[2]))
    shift = np.random.randint(-shift_max, shift_max + 1, size=(data.shape[0], 1, data.shape[2])) / 30.0
    return (data + noise) * scale + shift

data = augment_data(data)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels_onehot, test_size=0.2, stratify=labels_int, shuffle=True)

# Compute class weights for imbalance handling
class_weights = compute_class_weight('balanced', classes=np.unique(labels_int), y=labels_int)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Build optimized LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(30, 84)),  # Increased LSTM units
    BatchNormalization(),  # Normalization for stable training
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(len(label_to_int), activation='softmax')
])

# Compile model with Adam optimizer and learning rate scheduling
optimizer = Adam(learning_rate=0.001)  # Adam performs better for sequential data
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks: Early stopping + Reduce LR on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Train model with class weights
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), 
          callbacks=[early_stopping, reduce_lr], class_weight=class_weight_dict)

# Save model
model.save(r'C:\Users\harryy\Desktop\dghnisl\videoisl\lstm_model.h5')
print("✅ Model trained and saved successfully!")
