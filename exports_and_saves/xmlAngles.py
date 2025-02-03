import cv2
import time
import mediapipe as mp
import numpy as np
from math import acos, degrees
import xml.etree.ElementTree as ET

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

# Crear el elemento raíz del XML
root = ET.Element("HandAngles")

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

                # Crear un nuevo elemento XML para cada conjunto de ángulos
                hand_element = ET.SubElement(root, "Hand")
                ET.SubElement(hand_element, "Thumb").text = str(angle_thumb)
                ET.SubElement(hand_element, "Index").text = str(angle_index)
                ET.SubElement(hand_element, "Middle").text = str(angle_middle)
                ET.SubElement(hand_element, "Ring").text = str(angle_ring)
                ET.SubElement(hand_element, "Pinky").text = str(angle_pinky)
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Mostrar los ángulos en la esquina superior izquierda
        #cv2.putText(frame, f'Angulo Pulgar: {int(angle_thumb)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #cv2.putText(frame, f'Angulo Indice: {int(angle_index)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #cv2.putText(frame, f'Angulo Medio: {int(angle_middle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #cv2.putText(frame, f'Angulo Anular: {int(angle_ring)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #cv2.putText(frame, f'Angulo Menique: {int(angle_pinky)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Convertir el elemento raíz a un árbol XML y guardar en un archivo
tree = ET.ElementTree(root)
tree.write("angles_data.xml", encoding='utf-8', xml_declaration=True)
