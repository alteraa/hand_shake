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
    right_hand_color: tuple = (88, 205, 54)  # Green
    left_hand_color: tuple = (54, 88, 205)  # Blue
    finger_tips_color: tuple = (0, 0, 0)  # Black
    handshake_color: tuple = (0, 0, 255)  # Red
    margin: int = 10
    font_scale: float = 1
    font_thickness: int = 1

    def __post_init__(self):
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
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        detection_result = self.detector.detect(mp_image)
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
            self._draw_fingertips(frame, landmarks)
            self._draw_handshake(frame, landmarks)
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

    def _draw_hand_landmarks(self, image, landmarks, color):
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in landmarks]
        )

        solutions.drawing_utils.draw_landmarks(
            image,
            landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Add landmark numbers
        height, width, _ = image.shape
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.putText(
                image,
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
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    def _draw_hand_label(self, image, landmarks, label, color):
        height, width, _ = image.shape
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]

        text_x = int(min(x_coords) * width)
        text_y = int(min(y_coords) * height) - self.config.margin

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            self.config.font_scale,
            color,
            self.config.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_fingertips(self, image, landmarks):
        for i in [4, 8, 12, 16, 20]:
            x, y = (
                int(landmarks[i].x * image.shape[1]),
                int(landmarks[i].y * image.shape[0]),
            )
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1)

    def _draw_handshake(self, image, landmarks):
        # Draw a line connecting the fingertips of the two hands
        y_top = np.max([landmarks[4].y, landmarks[20].y])
        y_bottom = np.min([landmarks[4].y, landmarks[20].y])
        y_mid = landmarks[0].y
        if y_top > y_mid > y_bottom:
            height, width, _ = image.shape
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]

            text_x = int(min(x_coords) * width)
            text_y = int(min(y_coords) * height) - self.config.margin

            cv2.putText(
                image,
                "Handshake!",
                (text_x + 10, text_y + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                self.config.handshake_color,
                1,
                cv2.LINE_AA,
            )

def main():
    config = HandShakeDetectionAppConfig()
    app = HandShakeDetectionApp(config)
    app.start()


if __name__ == "__main__":
    main()
