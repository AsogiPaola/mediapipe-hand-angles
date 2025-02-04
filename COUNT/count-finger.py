import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates):
    centroid = np.mean(coordinates, axis=0)
    return int(centroid[0]), int(centroid[1])

def extract_coordinates(hand_landmarks, indices, width, height):
    return np.array([
        [int(hand_landmarks.landmark[i].x * width), int(hand_landmarks.landmark[i].y * height)]
        for i in indices
    ])

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicialización de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Índices de los puntos
thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Colores y nombres de los dedos
finger_info = [
    ("Pulgar", (180, 229, 255)),
    ("Indice", (128, 64, 128)),
    ("Medio", (0, 204, 255)),
    ("Anular", (48, 255, 48)),
    ("Menique", (192, 101, 21))
]

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        fingers_counter = "_"
        thickness = [2] * 5

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extracción de coordenadas
                coordinates_thumb = extract_coordinates(hand_landmarks, thumb_points, width, height)
                coordinates_palm = extract_coordinates(hand_landmarks, palm_points, width, height)
                coordinates_ft = extract_coordinates(hand_landmarks, fingertips_points, width, height)
                coordinates_fb = extract_coordinates(hand_landmarks, finger_base_points, width, height)

                # Calcular ángulo del pulgar
                p1, p2, p3 = coordinates_thumb
                l1, l2, l3 = np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3), np.linalg.norm(p1 - p2)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = angle > 150

                # Calcular centroide y distancias
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])

                d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                
                fingers = np.concatenate([[thumb_finger], d_centrid_ft > d_centrid_fb])
                fingers_counter = str(np.count_nonzero(fingers))

                for i, finger in enumerate(fingers):
                    if finger:
                        thickness[i] = -1

                # Dibujar landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Visualización
        cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)

        for i, (name, color) in enumerate(finger_info):
            x_start = 100 + i * 60
            cv2.rectangle(frame, (x_start, 10), (x_start + 50, 60), color, thickness[i])
            cv2.putText(frame, name, (x_start, 80), 1, 1, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
