import text2emotion as te


def process_text(text: str) -> dict:
    emotion = unify_results(te.get_emotion(text))
    return emotion


def unify_results(analysis: dict) -> dict:
    # Unify results to match
    new_analysis = {}
    for k, v in analysis.items():
        if k == "Happy":
            new_analysis["happy"] = v
        elif k == "Angry":
            new_analysis["angry"] = v
        elif k == "Surprise":
            new_analysis["surprise"] = v
        elif k == "Sad":
            new_analysis["sad"] = v
        elif k == "Fear":
            new_analysis["fear"] = v

    new_analysis["neutral"] = (new_analysis["happy"] + new_analysis["surprise"] +
                               new_analysis["sad"] + new_analysis["fear"]) / 4
    new_analysis["disgust"] = (
        new_analysis["angry"] + new_analysis["fear"]) / 2

    return new_analysis


def get_top_emotion(emotion_dict: dict) -> str:
    if emotion_dict == {}:
        return "None"
    else:
        return max(emotion_dict, key=emotion_dict.get)


if __name__ == "__main__":
    text = "I am so happy that you are here."
    analysis = process_text(text)
    print(f"Text Analysis: {analysis}")

    text = "I am so sad that you are here, I don't want to be here."
    analysis = process_text(text)
    print(f"Text Analysis: {analysis}")
