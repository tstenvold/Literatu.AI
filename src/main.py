import geocoder
import emotion_face
import emotion_text
import speech_recognition as sr
import pyttsx3
import time
import threading
import concurrent.futures
import os
from chatbot import chatbot


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
    face = {}
    chat = chatbot("knowledge.json")
    chat.set_location(get_location().city)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        response = chat.get_response()
        say_text(response)

        while True:
            if chat.state == 'mood':
                response = chat.get_response()
                say_text(response)

                face_event.clear()
                face_future = executor.submit(
                    emotion_face.stream_process_image, face_event)
                text = get_voice_input(recognizer)
                face_event.set()
                face = face_future.result()

                voice = emotion_text.process_text(text)

                print(f"Voice: {voice}")
                print(f"Face: {face}")

                # combine voice and face emotion and return the top emotion
                emotion = emotion_text.combine_emotions(voice, face)
                print(f"Combined emotion: {emotion}")
                chat.mood = emotion_text.get_top_emotion(emotion)
                print(f"The emotion is {chat.mood}")

            else:
                text = get_voice_input(recognizer)

            response = chat.get_response()

            print(f"You said: {text}")
            print(f"Bot says: {response}")
            say_text(response)

            if chat.state == 'goodbye':
                response = chat.get_response()
                say_text(response)
                break
            elif chat.state == 'recommendation':
                # TODO Generate recommendation here
                chat.set_recommendation(
                    "No Country For Old Men by Cormac McCarthy")


if __name__ == "__main__":
    main()
