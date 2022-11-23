import pyttsx3
import speech_recognition as sr
import emotion_text
import geocoder

recognizer = sr.Recognizer()


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


def main():

    location = get_location()
    say_text("Welcome! to the Literature A I app.")
    say_text("How are you feeling today?")

    text = get_voice_input()
    emotion = emotion_text.get_top_emotion(emotion_text.process_text(text))
    say_text("So you are feeling " + emotion + ".")


if __name__ == "__main__":
    main()
