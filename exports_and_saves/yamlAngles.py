import cv2
import time
import mediapipe as mp
import numpy as np
from math import acos, degrees
import yaml

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture("video.mp4")

def angle_calculate(coordinate, points, hand_landmarks, width, height):
    for index in points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinate.append([x, y])
        
    p1 = np.array(coordinate[0])
    p2 = np.array(coordinate[1])
    p3 = np.array(coordinate[2])

    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)

    # Calcular el ángulo
    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
    return angle

# PUNTOS DE LOS DEDOS SEGUN MEDIAPIPE
thumb_points = [1, 2, 4]  # Pulgar
index_points = [5, 6, 8]  # Dedo índice
middle_points = [9, 10, 12]  # Dedo medio
ring_points = [13, 14, 16]  # Dedo anular
pinky_points = [17, 18, 20]  # Dedo meñique

# Lista para almacenar los datos de los ángulos
angles_data = []

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while True:
        ret, frame = cap.read()
        time.sleep(0.01)
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                angle_thumb = angle_calculate([], thumb_points, hand_landmarks, width, height)
                angle_index = angle_calculate([], index_points, hand_landmarks, width, height)
                angle_middle = angle_calculate([], middle_points, hand_landmarks, width, height)
                angle_ring = angle_calculate([], ring_points, hand_landmarks, width, height)
                angle_pinky = angle_calculate([], pinky_points, hand_landmarks, width, height)

                # Agregar los ángulos al diccionario de datos
                angles_data.append({
                    "thumb": angle_thumb,
                    "index": angle_index,
                    "middle": angle_middle,
                    "ring": angle_ring,
                    "pinky": angle_pinky
                })
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Guardar los datos en un archivo YAML
with open("angles_data.yaml", "w") as yaml_file:
    yaml.dump(angles_data, yaml_file, default_flow_style=False)
