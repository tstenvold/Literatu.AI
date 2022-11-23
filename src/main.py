import pyttsx3


def main():
    engine = pyttsx3.init('sapi5')

    engine.setProperty('rate', 160)
    engine.say("Welcome to the Literature A I app.")
    engine.runAndWait()


if __name__ == "__main__":
    main()
