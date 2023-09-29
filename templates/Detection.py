import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# mediapipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained gesture classification model
model = tf.keras.models.load_model('/path/to/your/model')  # Substitua pelo caminho para o seu modelo

# Webcam init
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Hands coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))

                # Bounding box
                x_min = min(landmarks, key=lambda x: x[0])[0]
                x_max = max(landmarks, key=lambda x: x[0])[0]
                y_min = min(landmarks, key=lambda x: x[1])[1]
                y_max = max(landmarks, key=lambda x: x[1])[1]

                # Resize bounding box
                expansion_factor = 1.5
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - int(width * (expansion_factor - 1) / 2))
                x_max = min(frame.shape[1], x_max + int(width * (expansion_factor - 1) / 2))
                y_min = max(0, y_min - int(height * (expansion_factor - 1) / 2))
                y_max = min(frame.shape[0], y_max + int(height * (expansion_factor - 1) / 2))

                # Extract hand gesture image
                hand_gesture = frame[y_min:y_max, x_min:x_max]

                # Preprocess hand gesture image (resize to model input size)
                hand_gesture = cv2.resize(hand_gesture, (224, 224))
                hand_gesture = hand_gesture / 255.0  # Normalize
                hand_gesture = np.expand_dims(hand_gesture, axis=0)  # Add batch dimension

                # Classify hand gesture using the model
                prediction = model.predict(hand_gesture)


                probability_offensive = prediction[0][0]
                probability_non_offensive = 1 - probability_offensive

                # Definir a classe com base na probabilidade
                if probability_non_offensive > probability_offensive:
                    predicted_class = "Non-offensive"
                else:
                    predicted_class = "Offensive"

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Display class label
                cv2.putText(frame, predicted_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if predicted_class == 'Offensive':
                    # Gaussian blur 
                    roi = frame[y_min:y_max, x_min:x_max]
                    roi = cv2.GaussianBlur(roi, (121, 121), 0)
                    frame[y_min:y_max, x_min:x_max] = roi

        # Show resulting image
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()