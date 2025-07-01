import cv2
import mediapipe as mp
import numpy as np

class FocusTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Eye landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def detect_all(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        blink_detected = False
        direction = "focused"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mesh_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

            # EAR Calculation
            def EAR(eye):
                A = np.linalg.norm(mesh_points[eye[1]] - mesh_points[eye[5]])
                B = np.linalg.norm(mesh_points[eye[2]] - mesh_points[eye[4]])
                C = np.linalg.norm(mesh_points[eye[0]] - mesh_points[eye[3]])
                return (A + B) / (2.0 * C)

            left_ear = EAR(self.LEFT_EYE)
            right_ear = EAR(self.RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            # Blink detection
            if avg_ear < 0.19:  # You can adjust this threshold
                blink_detected = True

            # Eye Direction
            left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
            right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
            eye_center_x = (left_eye_x + right_eye_x) / 2
            if abs(eye_center_x - 0.5) > 0.08:
                direction = "distracted"

            return {
                "status": direction,
                "blink": blink_detected
            }

        return {
            "status": "no_face",
            "blink": False
        }



