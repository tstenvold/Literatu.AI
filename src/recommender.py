import json
import random
from fuzzywuzzy import fuzz


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
        loc_books = self.get_location_candidates(location)
        return self.generate_recommendation(loc_books)

    def get_mood_books(self, mood: str) -> list:
        return [book for book in self.books if mood == book['emotion']]

    def get_moods_candidates(self, moods: str) -> list:
        candidates = []
        for mood in moods:
            candidates.extend(self.get_mood_books(mood))

        return candidates

    def extract_first_author(self, book: list) -> str:
        if book is None or len(book) == 0:
            return None
        return book['authors'][0]['name']

    def lookup_book_by_title(self, title: str) -> list:
        book = max(self.books, key=lambda book: fuzz.ratio(
            book['title'], title))
        if len(book) == 0 or fuzz.partial_ratio(book['title'], title) <= 70:
            return None
        else:
            return book

    def lookup_author(self, author: str) -> list:
        a_book = max(self.books, key=lambda book: fuzz.ratio(
            book['authors'][0]['name'], author))
        if len(author) == 0 or fuzz.partial_ratio(a_book['authors'][0]['name'], author) <= 70:
            return None
        else:
            return a_book

    def get_candidates_from_previous(self, candidates: list) -> list:
        candidates = []
        for (title, rating) in self.previous_recommendation:
            if title is not None:
                ref = self.lookup_book_by_title(title)
                candidates.extend(self.get_similar_books(ref))

        return candidates

    def get_similar_books(self, ref: list) -> list:
        return [book for book in self.books if ref['title'] != book['title'] and len(set(book['similar_books']).intersection(set(ref['similar_books']))) >= 2]

    def generate_recommendation(self, candidates: list) -> list:
        if len(candidates) == 0:
            return None
        x = random.randint(0, len(candidates)-1)
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


if __name__ == '__main__':
    rec = Recommender()
    book = rec.get_current_mood_recommendation('happy')
    print(book['title'])
    similar = rec.get_similar_books(book)
    for book in similar:
        print("\t ", book['title'])
    title = rec.lookup_book_by_title('Great Gatsby')
    print(title['title'])
    author = rec.lookup_author('Dan brown')
    print(rec.extract_first_author(author))
