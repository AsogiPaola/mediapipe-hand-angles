import cv2
import mediapipe as mp
import yaml

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Conexiones de los brazos
arm_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
]

# Índices de las yemas de los dedos
finger_tips = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

#cap = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Lista para almacenar las coordenadas
data = {"arm_landmarks": [], "finger_tips": []}

with mp_pose.Pose(static_image_mode=False) as pose, mp_hands.Hands(static_image_mode=False) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa la detección de pose
        pose_results = pose.process(frame_rgb)

        # Procesa la detección de manos
        hands_results = hands.process(frame_rgb)

        frame_data = {"arms": [], "fingers": []}

        # Dibuja las conexiones de los brazos y guarda las coordenadas
        if pose_results.pose_landmarks:
            for connection in arm_connections:
                start_idx, end_idx = connection
                start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                end_landmark = pose_results.pose_landmarks.landmark[end_idx]

                # Almacena las coordenadas de los brazos
                frame_data["arms"].append({
                    "start": {"x": start_landmark.x, "y": start_landmark.y, "z": start_landmark.z},
                    "end": {"x": end_landmark.x, "y": end_landmark.y, "z": end_landmark.z}
                })

                # Dibuja los puntos y conexiones
                cv2.line(frame,
                         (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])),
                         (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])),
                         (128, 0, 250), 2)
                cv2.circle(frame,
                           (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])),
                           3, (255, 255, 255), -1)
                cv2.circle(frame,
                           (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])),
                           3, (255, 255, 255), -1)

        # Dibuja las manos y guarda las coordenadas de las yemas de los dedos
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                for finger_tip in finger_tips:
                    fingertip_landmark = hand_landmarks.landmark[finger_tip]
                    frame_data["fingers"].append({
                        "finger_tip": finger_tip.name,
                        "x": fingertip_landmark.x,
                        "y": fingertip_landmark.y,
                        "z": fingertip_landmark.z
                    })
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 128, 250), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        data["arm_landmarks"].append(frame_data["arms"])
        data["finger_tips"].append(frame_data["fingers"])

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()

# Guarda los datos en un archivo YAML
with open("saves/armshands.yaml", "w") as file:
    yaml.dump(data, file)
