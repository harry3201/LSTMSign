import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image

# Load trained LSTM model
MODEL_PATH = r'C:\Users\harryy\Desktop\dghnisl\videoisl\lstm_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Load label mappings
mapping_path = r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\label_mapping.pickle'
with open(mapping_path, 'rb') as f:
    label_mapping = pickle.load(f)

# Ensure correct dictionary structure
if 'translations' not in label_mapping or 'label_to_int' not in label_mapping:
    raise ValueError("❌ Error: label_mapping.pickle does not contain expected keys: 'translations' and 'label_to_int'")

label_translations = label_mapping['translations']  # Extract translations
gesture_classes = list(label_mapping['label_to_int'].keys())  # Ensure correct class order

# Load fonts (Increased font size for visibility)
hindi_font = ImageFont.truetype(r'C:\Users\harryy\Desktop\dghnisl\videoisl\fonts\NotoSansDevanagari-VariableFont_wdth,wght.ttf', 40)
gujarati_font = ImageFont.truetype(r'C:\Users\harryy\Desktop\dghnisl\videoisl\fonts\NotoSansGujarati-VariableFont_wdth,wght.ttf', 40)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Constants
SEQUENCE_LENGTH = 30
FEATURES_PER_HAND = 42  # 21 landmarks * (x, y)
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 2 hands (if both detected)
CONFIDENCE_THRESHOLD = 0.8  # Ignore predictions below this confidence

# Maintain a history of predictions for smoothing
sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=10)

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Real-time sign language inference started. Perform a gesture! (Press 'q' to exit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    keypoints = []

    # Check for detected hands
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2:
                break  # Consider only max 2 hands

            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])  # Only (x, y) to match training shape

        # Ensure correct padding when only one hand is detected
        while len(keypoints) < TOTAL_FEATURES:
            keypoints.extend([0] * (TOTAL_FEATURES - len(keypoints)))  # Pad to 84
    else:
        keypoints = [0] * TOTAL_FEATURES  # No hands detected
        prediction_history.clear()  # Clear predictions if no hands detected

    # Maintain sequence of last frames
    sequence.append(keypoints)

    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(list(sequence), axis=0)
        predictions = model.predict(input_data)

        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence >= CONFIDENCE_THRESHOLD:  # Only consider high-confidence predictions
            predicted_label = gesture_classes[predicted_index]
            prediction_history.append(predicted_label)

        if len(prediction_history) >= 5:  # Only consider stable predictions
            most_frequent_label = Counter(prediction_history).most_common(1)[0][0]

            # Ensure the label exists in translations
            if most_frequent_label not in label_translations:
                print(f"⚠️ Warning: '{most_frequent_label}' not found in translations! Skipping frame.")
                continue  # Skip this frame

            # Get translations
            translation = label_translations[most_frequent_label]

            # Convert OpenCV frame to PIL for font rendering
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Define text positions
            positions = [(50, 50), (50, 120), (50, 190)]
            colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255)]  # White, Yellow, Cyan
            fonts = [hindi_font, hindi_font, gujarati_font]
            texts = [
                translation.get('english', 'N/A'), 
                translation.get('hindi', 'N/A'), 
                translation.get('gujarati', 'N/A')
            ]

            # Draw text with black outline for better visibility
            for (x, y), text, font, color in zip(positions, texts, fonts, colors):
                # Outline (black)
                for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:  
                    draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))

                # Main text (brighter color)
                draw.text((x, y), text, font=font, fill=color)

            # Convert back to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Show video feed
    cv2.imshow('Real-Time Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
