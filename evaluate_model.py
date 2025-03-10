import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Paths
data_path = r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\data_sequences.pickle'
label_mapping_path = r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\label_mapping.pickle'

# Ensure label mapping file exists
if not os.path.exists(label_mapping_path):
    print("❌ Error: label_mapping.pickle not found. Please generate it first.")
    exit()

# Load label mapping
label_to_int = pickle.load(open(label_mapping_path, 'rb'))

# Load preprocessed data
data_dict = pickle.load(open(data_path, 'rb'))
data, labels = data_dict['data'], data_dict['labels']

# Convert labels using label_to_int mapping
labels = np.array([label_to_int[label] for label in labels])

# Convert data to NumPy array
data = np.array(data, dtype=np.float32)  # Convert to float32 for TensorFlow compatibility

# One-hot encode labels
labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_to_int))

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Load trained model
model = load_model(r'C:\Users\harryy\Desktop\dghnisl\videoisl\lstm_model.h5')

# ✅ Ensure x_test is a NumPy array
x_test = np.array(x_test, dtype=np.float32)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'✅ Test Accuracy: {accuracy * 100:.2f}%')
