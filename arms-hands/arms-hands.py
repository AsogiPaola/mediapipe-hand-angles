import cv2
import mediapipe as mp

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

#cap = cv2.VideoCapture("video_0002.mp4")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

        # Dibuja las conexiones de los brazos
        if pose_results.pose_landmarks:
            for connection in arm_connections:
                start_idx, end_idx = connection
                cv2.line(frame,
                         (int(pose_results.pose_landmarks.landmark[start_idx].x * frame.shape[1]),
                          int(pose_results.pose_landmarks.landmark[start_idx].y * frame.shape[0])),
                         (int(pose_results.pose_landmarks.landmark[end_idx].x * frame.shape[1]),
                          int(pose_results.pose_landmarks.landmark[end_idx].y * frame.shape[0])),
                         (128, 0, 250), 2)
                cv2.circle(frame,
                           (int(pose_results.pose_landmarks.landmark[start_idx].x * frame.shape[1]),
                            int(pose_results.pose_landmarks.landmark[start_idx].y * frame.shape[0])),
                           3, (255, 255, 255), -1)
                cv2.circle(frame,
                           (int(pose_results.pose_landmarks.landmark[end_idx].x * frame.shape[1]),
                            int(pose_results.pose_landmarks.landmark[end_idx].y * frame.shape[0])),
                           3, (255, 255, 255), -1)

        # Dibuja las manos
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 128, 250), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
