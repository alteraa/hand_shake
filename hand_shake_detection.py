import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from dataclasses import dataclass


@dataclass
class HandShakeDetectionAppConfig:
    hand_landmarker_path: str = "hand_landmarker.task"
    num_hands: int = 2
    width: int = 640
    height: int = 480
    milliseconds_per_frame: int = 250
    frame_title: str = "Hand Shake Detection"
    right_hand_color: tuple = (48, 205, 14)  # Green
    left_hand_color: tuple = (24, 40, 205)  # Blue
    finger_tips_color: tuple = (0, 0, 0)  # Black
    handshake_color: tuple = (0, 0, 255)  # Red
    margin: int = 10
    font_scale: float = 1
    font_thickness: int = 1
    # ! important variables
    max_shake_angle: float = 40
    normalize_to: float = 90

    def __post_init__(self):
        assert self.normalize_to > 0, "Normalize to must be greater than 0!"
        assert (
            self.max_shake_angle > 0 and self.max_shake_angle < self.normalize_to
        ), f"Max shake angle must be between 0 and {self.normalize_to}!"
        self.delay_ = 1 / self.milliseconds_per_frame


class HandShakeDetectionApp(threading.Thread):
    def __init__(self, config: HandShakeDetectionAppConfig):
        super().__init__()
        self.config = config
        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=self.config.hand_landmarker_path,
                ),
                num_hands=self.config.num_hands,
            )
        )

    def process_frame(self, frame):
        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        detection_result = self.detector.detect(mp_frame)
        if not detection_result.hand_landmarks:
            return frame

        for landmarks, handedness in zip(
            detection_result.hand_landmarks,
            detection_result.handedness,
        ):
            # Get color based on handedness
            label = handedness[0].category_name
            color = (
                self.config.right_hand_color
                if label == "Right"
                else self.config.left_hand_color
            )
            # run draw functions
            self._draw_hand_landmarks(frame, landmarks, color)
            self._draw_hand_label(frame, landmarks, label, color)
            self._draw_fingertips(frame, landmarks, self.config.finger_tips_color)
            self._draw_handshake(frame, landmarks, color)
        return frame

    def setup_cam(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

    def run(self):
        self.setup_cam()
        while True:
            time.sleep(self.config.delay_)
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to get frame!")
                break
            annotated_frame = self.process_frame(frame)
            cv2.imshow(self.config.frame_title, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_hand_landmarks(self, frame, landmarks, color):
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in landmarks]
        )

        solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Add landmark numbers
        height, width, _ = frame.shape
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.putText(
                frame,
                str(idx),
                (x - 10, y - 10),  # Offset text slightly above the point
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (0, 0, 0),
                1,  # Thickness
                cv2.LINE_AA,
            )

        # Draw bounding box
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        x_min = int(min(x_coords) * width)
        y_min = int(min(y_coords) * height)
        x_max = int(max(x_coords) * width)
        y_max = int(max(y_coords) * height)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    def _draw_hand_label(self, frame, landmarks, label, color):
        height, width, _ = frame.shape
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]

        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - self.config.margin

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            self.config.font_scale,
            color,
            self.config.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_fingertips(self, frame, landmarks, color):
        for i in [4, 8, 12, 16, 20]:
            x, y = (
                int(landmarks[i].x * frame.shape[1]),
                int(landmarks[i].y * frame.shape[0]),
            )
            cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)

    def _draw_handshake(self, frame, landmarks, color):
        # Draw a line connecting the fingertips of the two hands
        y_top = np.max([landmarks[4].y, landmarks[20].y])
        y_bottom = np.min([landmarks[4].y, landmarks[20].y])
        y_mid = landmarks[0].y
        # Calculate the angle of the line connecting the palm and two knuckles
        p0 = np.asarray([landmarks[0].x, landmarks[0].y])  # Wrist
        p5 = np.asarray([landmarks[5].x, landmarks[5].y])  # Index knuckle
        p17 = np.asarray([landmarks[17].x, landmarks[17].y])  # Pinky knuckle
        v1 = p17 - p0
        v2 = p5 - p17
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        th = (1 - np.abs(cos)) * self.config.normalize_to

        is_handshake = (y_top > y_mid > y_bottom) and (th < self.config.max_shake_angle)

        # drawing the angle and the handshake text
        height, width, _ = frame.shape
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - self.config.margin
        cv2.putText(
            frame,
            f"{th:.2f}",
            (text_x + 40, text_y - 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            color,  # light gray
            1,
            cv2.LINE_AA,
        )
        if is_handshake:
            cv2.putText(
                frame,
                "Handshake",
                (text_x + 20, text_y + 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                self.config.handshake_color,
                1,
                cv2.LINE_AA,
            )


def main():
    config = HandShakeDetectionAppConfig(
        max_shake_angle=40,
        normalize_to=90,
    )
    app = HandShakeDetectionApp(config)
    app.start()


if __name__ == "__main__":
    main()
