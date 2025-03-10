import os
import cv2
import numpy as np
import pickle
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Users\harryy\Desktop\dghnisl\videoisl\videodata'
PREPROCESSED_DATA_DIR = r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data'
if not os.path.exists(PREPROCESSED_DATA_DIR):
    os.makedirs(PREPROCESSED_DATA_DIR)

FEATURE_VECTOR_LENGTH = 42
MAX_HANDS = 2
SEQUENCE_LENGTH = 30

data = []
labels = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    print(f"Processing class: {class_name}")
    for sequence_folder in os.listdir(class_path):
        sequence_path = os.path.join(class_path, sequence_folder)
        if not os.path.isdir(sequence_path):
            continue
        print(f"Processing sequence folder: {sequence_folder}")
        frames = []
        for frame_file in sorted(os.listdir(sequence_path)):
            frame_path = os.path.join(sequence_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to load frame: {frame_path}")
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            data_aux = []
            x_ = []
            y_ = []

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= MAX_HANDS:
                        break

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                if len(results.multi_hand_landmarks) < MAX_HANDS:
                    padding = [0] * FEATURE_VECTOR_LENGTH
                    data_aux.extend(padding)

            if len(data_aux) == FEATURE_VECTOR_LENGTH * MAX_HANDS:
                frames.append(data_aux)

        if len(frames) == SEQUENCE_LENGTH:
            data.append(frames)
            labels.append(class_name)
            print(f"Added sequence for class: {class_name}")

print(f"Data shape: {np.array(data).shape}")
print(f"Labels: {labels}")

# Convert labels to integers
label_to_int = {label: idx for idx, label in enumerate(set(labels))}
print(f"Label to int mapping: {label_to_int}")

labels = np.array([label_to_int[label] for label in labels])
print(f"Labels after conversion: {labels}")

# Save preprocessed data
pickle.dump({'data': data, 'labels': labels}, open(os.path.join(PREPROCESSED_DATA_DIR, 'data_sequences.pickle'), 'wb'))