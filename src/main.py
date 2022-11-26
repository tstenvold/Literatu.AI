import geocoder
import emotion_face
import emotion_text
import speech_recognition as sr
import pyttsx3
import time
import threading
import concurrent.futures
import os
from chatbot import Chat


def get_location() -> str:
    g_loc = geocoder.ip('me')
    print(f"Current city is {g_loc.city}")
    return g_loc


def say_text(text: str):

    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()


def get_voice_input(recognizer) -> str:
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            recognizer.pause_threshold = 0.5
            audio = recognizer.listen(source)
            out_text = recognizer.recognize_google(audio)
            out_text = out_text.lower()
            return out_text

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")
        time.sleep(5)
        return ""


def main():

    recognizer = sr.Recognizer()
    face_event = threading.Event()
    text = ""
    face = {}
    chat = Chat()
    chat.start_new_session(1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(get_location)
        # say_text("Welcome! to the Literature A I app.")
        first_q = "How are you feeling today?"
        chat._conversation[1].append(first_q)
        say_text(first_q)

        while True:
            face_future = executor.submit(
                emotion_face.stream_process_image, face_event)
            text = get_voice_input(recognizer)
            face_event.set()
            face = face_future.result()

            voice = emotion_text.process_text(text)
            chat._conversation[1].append(text)

            print(f"Voice: {voice}")
            print(f"Face: {face}")

            voice_emotion = emotion_text.get_top_emotion(voice)
            face_emotion = emotion_face.get_top_emotion(face)

            print(f"Voice Emotion: {voice_emotion}")
            print(f"Face Emotion: {face_emotion}")

            response = chat.respond(text, 1)
            chat._conversation[1].append(response)

            print(f"You said: {text}")
            print(f"Bot says: {response}")
            say_text(response)

            if text in ["quit", "exit", "bye", "cancel", "stop"]:
                break


if __name__ == "__main__":
    main()
