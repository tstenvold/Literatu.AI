import json
import random


class Recommender:

    def __init__(self, file_path='random_sample_final_short.json'):
        self.books = []
        self.previous_recommendation = None

        with open(file_path, 'r') as fp:
            self.books = json.load(fp)

    def load_previous_recommendation(self, recommendation: list, rating: int):
        self.previous_recommendation = zip(recommendation, rating)
        self.remove_previous_recommendation()

    def remove_previous_recommendation(self):
        for (title, _) in self.previous_recommendation:
            if title is not None:
                self.books = [
                    book for book in self.books if book['title'] != title]

    def get_location_candidates(self, location: str) -> list:
        return [book for book in self.books if location in book['precise_country']]

    def get_location_recommendation(self, location: str) -> list:
        # Return a random book from the location
        loc_books = self.get_location_candidates(location)
        return self.generate_recommendation(loc_books)

    def get_mood_books(self, mood: str) -> list:
        return [book for book in self.books if mood == book['emotion']]

    def get_moods_candidates(self, moods: str) -> list:
        candidates = []
        for mood in moods:
            candidates.extend(self.get_mood_books(mood))

        return candidates

    def generate_recommendation(self, candidates: list) -> list:
        if len(candidates) == 0:
            return None
        x = random.randint(0, len(candidates)-1)
        print(f"Recommendation: ", candidates[x])
        return candidates[x]

    def get_current_mood_recommendation(self, user_mood: str) -> list:
        if user_mood == 'happy' or user_mood == 'neutral' or user_mood == 'surprise':
            moods = ['happy', 'neutral', 'sad', 'angry', 'fear', 'surprise']
        elif user_mood == 'sad':
            moods = ['happy', 'surprise']
        elif user_mood == 'angry':
            moods = ['happy', 'surprise', 'neutral']
        else:
            moods = ['happy', 'neutral']

        candidates = self.get_moods_candidates(moods)
        return self.generate_recommendation(candidates)
