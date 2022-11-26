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
    elif emotion_dict["happy"] == emotion_dict["angry"] == emotion_dict["surprise"] == emotion_dict["sad"] == emotion_dict["fear"] == emotion_dict["neutral"] == emotion_dict["disgust"]:
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
        results.append(process_frame(get_image(cap)))
        count += 1
        if count >= 5:
            break

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
