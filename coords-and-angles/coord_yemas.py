import cv2
import mediapipe as mp
import yaml

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Diccionario con los nombres de los dedos
finger_names = {
    4: "Pulgar",
    8: "Indice",
    12: "Medio",
    16: "Anular",
    20: "Menique"
}

# Lista para almacenar las coordenadas de los dedos
all_finger_coords = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja los puntos y las conexiones en la imagen
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=4, circle_radius=5))

                finger_coords = {}
                for tip_id, finger_name in finger_names.items():
                    # Obten las coordenadas X y Y de las yemas de los dedos
                    x = int(hand_landmarks.landmark[tip_id].x * width)
                    y = int(hand_landmarks.landmark[tip_id].y * height)
                    finger_coords[finger_name] = {'X': x, 'Y': y}

                # Añadir coordenadas actuales a la lista
                all_finger_coords.append(finger_coords)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Guardar las coordenadas en un archivo YAML
with open('coordenadas.yaml', 'w') as file:
    yaml.dump(all_finger_coords, file)

cap.release()
cv2.destroyAllWindows()
