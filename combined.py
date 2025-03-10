import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# 🎯 Load Static Gesture Model (Random Forest)
with open('model.p', 'rb') as f:
    static_model = pickle.load(f)['model']

# 🎯 Load Labels from `data.pickle`
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
gesture_classes = data['labels']

# 🎯 Load Dynamic Gesture Model (LSTM)
dynamic_model = load_model('lstm_model.h5')

# 🎯 Load Fonts for Translations
font_hindi = ImageFont.truetype('fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf', 40)
font_gujarati = ImageFont.truetype('fonts/NotoSansGujarati-VariableFont_wdth,wght.ttf', 40)
font_prediction = ImageFont.truetype("arial.ttf", 40)

# 🎥 Initialize Webcam
cap = cv2.VideoCapture(0)

# ✋ Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# ⏳ Motion Tracking Variables
prev_landmarks = None
MOTION_THRESHOLD = 0.005  # Adjusted for better motion detection

# ⏳ Sequence Buffer for LSTM
sequence = []
SEQUENCE_LENGTH = 30  # Number of frames for LSTM prediction
prediction_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    keypoints = []
    motion = 0
    predicted_label = "Unknown"
    confidence = 0  # Confidence score

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        hand_points = []

        for hand_landmarks in results.multi_hand_landmarks:
            temp_points = []
            for landmark in hand_landmarks.landmark:
                x, y = landmark.x, landmark.y
                temp_points.append((x, y))
                keypoints.extend([x, y])

            hand_points.append(temp_points)

        # Ensure we have exactly 84 features (42 landmarks × 2)
        if num_hands == 1:
            keypoints.extend([0.0] * (84 - len(keypoints)))  # Pad missing values

        # ✅ Debug: Print keypoints to check if they're detected
        print(f"Keypoints Extracted: {len(keypoints)} values -> {keypoints[:10]} ...")  

        # ✅ Motion Detection
        if prev_landmarks and len(prev_landmarks) == len(hand_points[0]):
            motion = sum(
                np.linalg.norm(np.array(p1) - np.array(p2))
                for p1, p2 in zip(prev_landmarks, hand_points[0])
            )
        prev_landmarks = hand_points[0]

        print(f"Motion Value: {motion}")  # ✅ Debug Motion

        # 🎯 **Static Gesture Prediction (A-Z, 0-9)**
        if motion < MOTION_THRESHOLD and len(keypoints) == 84:
            static_features = np.array([keypoints])
            static_prediction = static_model.predict(static_features)[0]

            print(f"Static Model Prediction: {static_prediction}")  # ✅ Debug Static Prediction

            if 0 <= int(static_prediction) < len(gesture_classes):
                predicted_label = gesture_classes[int(static_prediction)]
                confidence = 1.0  # Static gestures have full confidence (RF model)
            else:
                print(f"❌ Warning: Invalid index {static_prediction}")

    # 🎯 **Dynamic Gesture Prediction (LSTM-based)**
    if motion >= MOTION_THRESHOLD and len(keypoints) == 84:
        sequence.append(keypoints)
        print(f"Sequence Length: {len(sequence)}")  # ✅ Debug sequence filling up

        if len(sequence) == SEQUENCE_LENGTH:
            sequence_input = np.expand_dims(sequence, axis=0)
            softmax_scores = dynamic_model.predict(sequence_input)[0]  # Get softmax output
            dynamic_prediction = np.argmax(softmax_scores)
            confidence = softmax_scores[dynamic_prediction]  # Confidence of predicted class

            print(f"Dynamic Model Prediction: {dynamic_prediction}, Confidence: {confidence:.4f}, Scores: {softmax_scores}")  # ✅ Debug Dynamic Prediction

            if confidence < 0.5:
                print(f"⚠️ Low Confidence in Prediction: {dynamic_prediction}, Score: {confidence:.4f}")  # ✅ Log low-confidence cases

            predicted_label = f"Dynamic-{dynamic_prediction}"  # Replace with actual class mapping
            sequence = []  # Reset buffer

    # ✅ Save Prediction History for Stability
    prediction_history.append(predicted_label)
    if len(prediction_history) > 5:
        prediction_history.pop(0)

    # ✅ Display Prediction
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 20), f"Prediction: {predicted_label} ({confidence:.2f})", font=font_prediction, fill=(0, 255, 0))

    frame = np.array(img_pil)
    cv2.imshow('Sign Language Recognition', frame)

    # 🛑 Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
