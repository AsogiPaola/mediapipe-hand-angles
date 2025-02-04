#this code is for measure between 4 fingers ignoring thumb
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializa la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configura MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja las conexiones de los puntos de referencia
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=4, circle_radius=5))

                # Define los índices de las yemas de los dedos
                landmarks_indices = [4, 8, 12, 16, 20]  # Pulgar, índice, medio, anular, meñique
                
                # Obtén las coordenadas de las yemas de los dedos
                landmarks = np.array([[hand_landmarks.landmark[i].x * width,
                                       hand_landmarks.landmark[i].y * height]
                                      for i in landmarks_indices])
                
                # Calcula las distancias entre las yemas de los dedos
                distances = [np.linalg.norm(landmarks[i] - landmarks[i + 1])
                             for i in range(len(landmarks) - 1)]
                
                # Muestra las distancias calculadas en la consola
                #print(f'Distancias: {distances}')
                
                # muestra las distancias en la ventana de video
                for i, (start_idx, end_idx) in enumerate(zip(landmarks_indices[:-1], landmarks_indices[1:])):
                    start = tuple(np.round(landmarks[i]).astype(int))
                    end = tuple(np.round(landmarks[i + 1]).astype(int))
                    cv2.putText(frame, f'{distances[i]:.2f}', 
                                ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Muestra la imagen procesada
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
            break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
