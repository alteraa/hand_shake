import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from dataclasses import dataclass


@dataclass
class HandShakeRecognitionAppConfig:
    gesture_recognizer_path: str = "gesture_recognizer.task"
    num_hands: int = 2
    width: int = 640
    height: int = 480
    milliseconds_per_frame: int = 250
    frame_title: str = "Hand Shake Detection"
    right_hand_color: tuple = (48, 205, 14)  # Green
    left_hand_color: tuple = (205, 40, 24)  # Blue
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


class HandShakeRecognitionApp(threading.Thread):
    def __init__(self, config: HandShakeRecognitionAppConfig):
        super().__init__()
        self.config = config
        with open(self.config.gesture_recognizer_path, "rb") as f:
            self.recognizer = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(
                    base_options=python.BaseOptions(model_asset_buffer=f.read()),
                    num_hands=self.config.num_hands,
                )
            )

    def process_frame(self, frame):
        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        recognition_result = self.recognizer.recognize(mp_frame)
        if not recognition_result.hand_landmarks:
            return frame

        for landmarks, handedness, gestures in zip(
            recognition_result.hand_landmarks,
            recognition_result.handedness,
            recognition_result.gestures,
        ):
            # Get color based on handedness
            hand_label = handedness[0].category_name
            color = (
                self.config.right_hand_color
                if hand_label == "Right"
                else self.config.left_hand_color
            )
            self._draw_hand(
                frame=frame,
                landmarks=landmarks,
                gesture_label=gestures[0].category_name,
                color=color,
                hand_label=hand_label,
                show_hand_label=True,
                show_landmark_numbers=True,
                show_bbox=True,
                show_fingertips=True,
            )
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

    def _draw_hand(
        self,
        frame,
        landmarks,
        gesture_label,
        color,
        hand_label=None,
        show_hand_label=False,
        show_landmark_numbers=False,
        show_bbox=False,
        show_fingertips=False,
    ):
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
        height, width, _ = frame.shape
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        x_gesture_label = int(min(x_coords) * width)
        y_gesture_label = int(min(y_coords) * height) - self.config.margin

        # show gesture
        cv2.putText(
            frame,
            gesture_label,
            (x_gesture_label, y_gesture_label),
            cv2.FONT_HERSHEY_DUPLEX,
            self.config.font_scale,
            (0, 0, 255),
            self.config.font_thickness,
            cv2.LINE_AA,
        )

        if show_hand_label:
            cv2.putText(
                frame,
                hand_label,
                (x_gesture_label + 40, y_gesture_label - 30),
                cv2.FONT_HERSHEY_DUPLEX,
                self.config.font_scale,
                color,
                self.config.font_thickness,
                cv2.LINE_AA,
            )

        if show_landmark_numbers:
            # Add landmark numbers
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

        if show_bbox:
            x_min = int(min(x_coords) * width)
            y_min = int(min(y_coords) * height)
            x_max = int(max(x_coords) * width)
            y_max = int(max(y_coords) * height)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        if show_fingertips:
            for i in [4, 8, 12, 16, 20]:
                x, y = (
                    int(landmarks[i].x * frame.shape[1]),
                    int(landmarks[i].y * frame.shape[0]),
                )
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)


def main():
    config = HandShakeRecognitionAppConfig(
        gesture_recognizer_path="handshake_recognizer.task",
        max_shake_angle=40,
        normalize_to=90,
    )
    app = HandShakeRecognitionApp(config)
    app.start()


if __name__ == "__main__":
    main()
