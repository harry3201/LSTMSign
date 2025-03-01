import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r'C:\Users\harryy\Desktop\dghnisl\videoisl\videodata'
SEQUENCE_LENGTH = 30
FEATURE_VECTOR_LENGTH = 42
MAX_HANDS = 2

data = []
labels = []

for label in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, label)
    
    for seq in os.listdir(class_path):
        sequence_data = []
        
        for frame_idx in range(SEQUENCE_LENGTH):
            img_path = os.path.join(class_path, seq, f'{frame_idx}.jpg')
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            keypoints = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y])  # Extract (x, y)

                if len(results.multi_hand_landmarks) == 1:
                    keypoints.extend([0] * FEATURE_VECTOR_LENGTH)  # Pad if one hand

            else:
                keypoints = [0] * (FEATURE_VECTOR_LENGTH * MAX_HANDS)

            sequence_data.append(keypoints)

        if len(sequence_data) == SEQUENCE_LENGTH:
            data.append(sequence_data)
            labels.append(label)

# Save dataset
with open(r'C:\Users\harryy\Desktop\dghnisl\videoisl\preprocessed_data\data_sequences.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
