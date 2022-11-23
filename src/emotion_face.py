from fer import FER
from collections import Counter
import time
import os
import cv2


def get_num_images(num_image: int, interval: int = 100) -> None:
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    for n in range(num_image):
        time.sleep(interval / 1000)
        ret, frame = cap.read()
        if frame is not None:
            cv2.imwrite(f"assets/images/cam_emo_{n}.png", frame)
        else:
            n -= 1

    cap.release()


def process_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    emotion_detector = FER(mtcnn=True)
    analysis = emotion_detector.detect_emotions(img).pop()
    return analysis.get('emotions')


def get_top_emotion(emotion_dict: dict) -> str:
    return max(emotion_dict, key=emotion_dict.get)


def determine_emotion() -> str:

    analysis = {}
    num_images = 5

    get_num_images(num_images)

    for i in range(5):
        if analysis == {}:
            analysis = process_image(f"assets/images/cam_emo_{i}.png")
        else:
            dict(Counter(analysis) +
                 Counter(process_image(f"assets/images/cam_emo_{i}.png")))

    return {k: v / num_images for k, v in analysis.items()}


if __name__ == "__main__":
    sample = process_image("assets/images/sample_image.png")
    print(f"Sample Image Analysis: {sample}")
    print(f"Top Emotion: {get_top_emotion(sample)}")

    emotion_dict = determine_emotion()
    print(f"Top Emotion: {get_top_emotion(emotion_dict)}")

    for i in range(5):
        os.remove(f"assets/images/cam_emo_{i}.png")
