import os
import cv2

# Define the dataset directory
DATA_DIR = r'C:\Users\harryy\Desktop\dghnisl\videoisl\videodata'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 3  # Update based on classes (ball, eat, book)
dataset_size = 50  # Number of sequences per class
sequence_length = 30  # Number of frames per sequence

# Initialize webcam
cap = cv2.VideoCapture(0)

for _ in range(number_of_classes):
    class_name = input("Enter the name of the gesture class (e.g., 'ball'): ")
    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    print(f'Collecting data for class: {class_name}')

    recording = False
    paused = False

    # Wait for user to start recording
    while not recording:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(frame, 'Press "Z" to start recording!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('z'):
            recording = True  # Start recording

    if not recording:
        continue  # Skip to next class if recording didn't start

    # Collect the required number of sequences
    for seq in range(dataset_size):
        sequence_folder = os.path.join(class_path, f'seq_{seq}')
        os.makedirs(sequence_folder, exist_ok=True)

        print(f"📸 Collecting sequence {seq+1}/{dataset_size} for {class_name}")

        # Capture frames for each sequence
        for frame_idx in range(sequence_length):
            while paused:
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):  # Resume on 'q' key
                    paused = False  

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_path = os.path.join(sequence_folder, f'{frame_idx}.jpg')
            cv2.imwrite(frame_path, frame)  # Save the frame

            cv2.putText(frame, f"Recording {class_name}: {seq+1}/{dataset_size}, Frame {frame_idx+1}/{sequence_length}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):  
                paused = True  # Pause recording
            elif key == ord('z'):  
                print(f"❌ Stopping recording for {class_name}")
                recording = False
                break  # Stop collecting sequences for this class

        if not recording:
            break  # Stop collecting sequences for this class

cap.release()
cv2.destroyAllWindows()
