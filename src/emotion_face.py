import random
import concurrent.futures
import string
import threading
from fer import FER
from collections import Counter
import time
import os
import cv2


def random_string(length: int) -> list:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def get_image(cap):

    while True:
        time.sleep(0.25)
        _, frame = cap.read()
        if frame is not None:
            return frame


def process_frame(frame) -> dict:
    emotion_detector = FER(mtcnn=True)
    try:
        analysis = emotion_detector.detect_emotions(frame)
        result = {}
        if analysis.__len__() != 0:
            analysis = analysis.pop()
            result = analysis.get('emotions')

        return result
    except Exception as e:
        return {}


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
    analysis = emotion_detector.detect_emotions(img)
    result = {}
    if analysis.__len__() != 0:
        analysis = analysis.pop()
        result = analysis.get('emotions')

    return result


def get_top_emotion(emotion_dict: dict) -> str:
    if emotion_dict == {}:
        return "None"
    else:
        return max(emotion_dict, key=emotion_dict.get)


def stream_process_image(event: threading.Event) -> None:
    results = []
    count = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    while not event.is_set():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(process_frame, get_image(cap))
            results.append(future.result())
        count += 1
        time.sleep(0.1)

    cap.release()

    final_analysis = combine_queued_results(results)
    return final_analysis


def combine_queued_results(results) -> dict:
    count = 0
    analysis = {}
    for r in results:
        if analysis == {}:
            analysis = r
        else:
            dict(Counter(analysis) + Counter(r))
        count += 1

    final_analysis = {k: v / count for k, v in analysis.items()}
    return final_analysis


def determine_emotion() -> dict:

    analysis = {}
    num_images = 5

    get_num_images(num_images)

    for i in range(5):
        if analysis == {}:
            analysis = process_image(f"assets/images/cam_emo_{i}.png")
        else:
            dict(Counter(analysis) +
                 Counter(process_image(f"assets/images/cam_emo_{i}.png")))

    final_analysis = {k: v / num_images for k, v in analysis.items()}
    print(f"Analysis: {final_analysis}")
    return final_analysis


if __name__ == "__main__":
    sample = process_image("assets/images/sample_image.png")
    print(f"Sample Image Analysis: {sample}")
    print(f"Top Emotion: {get_top_emotion(sample)}")

    emotion_dict = determine_emotion()
    print(f"Top Emotion: {get_top_emotion(emotion_dict)}")
