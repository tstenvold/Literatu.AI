import concurrent.futures
import threading
import time
import pyttsx3
import speech_recognition as sr
import emotion_text
import emotion_face
import geocoder


def get_location() -> str:
    g_loc = geocoder.ip('me')
    print(f"Current city is {g_loc.city}")
    return g_loc


def say_text(text: str):

    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()


def get_voice_input() -> str:
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            audio = recognizer.listen(source)
            out_text = recognizer.recognize_google(audio)
            out_text = out_text.lower()
            return out_text

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")
        time.sleep(5)
        return "Generic text, with no emotions"


def main():

    location = get_location()
    say_text("Welcome! to the Literature A I app.")
    say_text("How are you feeling today?")

    event = threading.Event()
    text = ""
    face = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(emotion_face.stream_process_image, event)
        text = get_voice_input()
        event.set()
        face = future.result()

    voice = emotion_text.process_text(text)
    
    print(f"Voice: {voice}")
    print(f"Face: {face}")

    voice_emotion = emotion_text.get_top_emotion(voice)
    face_emotion = emotion_face.get_top_emotion(face)

    say_text("So you are feeling " + voice_emotion + ".")
    print(f"Voice Emotion: {voice_emotion}")

    say_text(", and your face is showing " + face_emotion + ".")
    print(f"Face Emotion: {face_emotion}")


if __name__ == "__main__":
    main()
