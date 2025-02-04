import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def angle_calculate(coordinate, points):
    for index in points:
        x = int(hand_landmarks.landmark[index].x * width)
        y = int(hand_landmarks.landmark[index].y * height)
        coordinate.append([x, y])
        
    p1 = np.array(coordinate[0])
    p2= np.array(coordinate[1])
    p3= np.array(coordinate[2])

    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)

    # Calcular el ángulo
    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
    return angle                 

#PUNTOS DE LOS DEDOS SEGUN MEDIAPIPE
thumb_points = [1, 2, 4] # Pulgar
index_points = [5, 6, 8] # dedo indice
middle_points = [9, 10, 12] #dedo middle
ring_points = [13, 14, 16] #dedo ring
pinky_points = [17, 18, 20] #dedo pinky

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    # Inicialización de variables
    angle_thumb = angle_index = angle_middle = angle_ring = angle_pinky = 0
    #finger_name = ["Pulgar", "Indice", "Medio", "Anular", "Meñique"]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        thickness = [2, 2, 2, 2, 2]

        if results.multi_hand_landmarks:
            """coordinates_thumb = []
            coordinates_index = []
            coordinates_middle = []
            coordinates_ring = []
            coordinates_pinky = []"""
           
            for hand_landmarks in results.multi_hand_landmarks:
                angle_thumb = angle_calculate([], thumb_points)                
                angle_index = angle_calculate([], index_points)                
                angle_middle = angle_calculate([], middle_points)
                angle_ring = angle_calculate([], ring_points)
                angle_pinky = angle_calculate([], pinky_points)
                
                if angle_thumb and angle_middle and angle_index and angle_ring and angle_pinky < 90:
                    cv2.putText(frame, f'MANO CERRADA', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Mostrar el ángulo en la esquina superior izquierda
        cv2.putText(frame, f'Angulo Pulgar: {int(angle_thumb)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Angulo Indice: {int(angle_index)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Angulo Medio: {int(angle_middle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Angulo Anular: {int(angle_ring)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Angulo Menique: {int(angle_pinky)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
