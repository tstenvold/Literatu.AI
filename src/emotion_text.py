import text2emotion as te
import nltk
nltk.download('omw-1.4')


def process_text(text: str) -> dict:
    emotion = te.get_emotion(text)
    return emotion


if __name__ == "__main__":
    text = "I am so happy that you are here."
    analysis = process_text(text)
    print(f"Text Analysis: {analysis}")

    text = "I am so sad that you are here, I don't want to be here."
    analysis = process_text(text)
    print(f"Text Analysis: {analysis}")
