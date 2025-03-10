import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load preprocessed sequences
data_dict = pickle.load(open(r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\data_sequences.pickle', 'rb'))
data = np.array(data_dict['data'])  # Shape: (num_samples, 30, 84)
labels = data_dict['labels']

# Define the correct label order
ordered_labels = ["ball", "book", "clean", "cold", "deny", "grateful", "hurt", "milk", "playing games", "rice"]

# Create label-to-index mapping
label_to_int = {label: idx for idx, label in enumerate(ordered_labels)}

# Label translations (for displaying predictions)
label_translations = {
    "ball": {"english": "Ball", "hindi": "गेंद", "gujarati": "બોલ"},
    "book": {"english": "Book", "hindi": "पुस्तक", "gujarati": "પુસ્તક"},
    "clean": {"english": "Clean", "hindi": "साफ", "gujarati": "સાફ"},
    "cold": {"english": "Cold", "hindi": "ठंडा", "gujarati": "ઠંડો"},
    "deny": {"english": "Deny", "hindi": "इंकार", "gujarati": "નકાર"},
    "grateful": {"english": "Grateful", "hindi": "आभारी", "gujarati": "આભારી"},
    "hurt": {"english": "Hurt", "hindi": "चोट", "gujarati": "ઈજા"},
    "milk": {"english": "Milk", "hindi": "दूध", "gujarati": "દૂધ"},
    "playing games": {"english": "Playing Games", "hindi": "खेल खेलना", "gujarati": "રમત રમી"},
    "rice": {"english": "Rice", "hindi": "चावल", "gujarati": "ચોખા"}
}

# Save both label-to-index and translations in one file
label_mapping = {"label_to_int": label_to_int, "translations": label_translations}
with open(r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\label_mapping.pickle', 'wb') as f:
    pickle.dump(label_mapping, f)

# Convert labels to integers
labels_int = np.array([label_to_int[label] for label in labels])

# One-hot encode labels
labels_onehot = to_categorical(labels_int, num_classes=len(label_to_int))

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_onehot, test_size=0.2, stratify=labels_int, shuffle=True
)

print("✅ Data preprocessing complete! Label mapping updated.")
