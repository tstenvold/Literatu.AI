import json
import random
import os
import string
import pycountry
from transformers import pipeline


class chatbot:

    def __init__(self, knowledge_json=None) -> None:
        self.state = 'greeting'
        self.last_state = None
        self.num_questions = 2

        self.nlp = pipeline(
            'question-answering',
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.summarize = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            tokenizer="sshleifer/distilbart-cnn-12-6"
        )

        self.mood = None
        self.genre = None
        self.author = None
        self.book = None
        self.location = None
        self.last_book = None
        self.rating = None
        self.recommendation = None
        self.recommendation_summary = None
        self.rec = None

        self.cancel_words = ["stop", "cancel", "quit", "exit", ]
        self.responses = {}

        self.knowledge = {}
        if knowledge_json:
            self.load_knowledge(knowledge_json)

        self.update_responses()
        # Print all the loaded variables
        print(f"genre: {self.genre}")
        print(f"author: {self.author}")
        print(f"book: {self.book}")

    def update_responses(self) -> None:
        self.responses = {
            'greeting': ['Hello', 'Hi', 'Welcome', 'Good day', ],
            'confirm': ['OK', 'alright', 'I see', 'I get it', 'interesting', ],
            'mood': ['How are you?', 'How are you doing?', 'How are you feeling?', 'How are you today?'],
            'genre': ['What genre do you like?', 'What genre do you prefer?', 'What genre do you enjoy?', 'What genre do you read?'],
            'author': ['Who is your favorite author?', 'Which author do you like?', 'Name an author that you like'],
            'book': ['What is your favorite book?', 'Which book have you liked?', 'Name a book that you like'],
            'location': [f'Would you like to read a book from {self.get_country_name()}?', f'Would you like to read a {self.get_country_name()} book?'],
            'last_book': [f'Did you finish {self.last_book}?', f'Did you complete {self.last_book}?', f'Have you finished {self.last_book}?'],
            'rating': ['How would you rate it from 0 to 10?', 'What would you rate it out of 10?', 'Please rate it from 0 to 10'],
            'recommendation': [f'I recommend you read {self.recommendation}', f'I think you should read {self.recommendation}', f'I suggest you look into {self.recommendation}', f'I think you would like {self.recommendation}'],
            'goodbye': ['Goodbye', 'See you later', 'Bye', 'Have a nice day', 'Have a good day'],
            'new_book': ['Would you like to read a different book?', 'Shall I recommend you a different book?'],
            'summary': ['Would you like to hear a summary?', 'Do you want to hear a brief summary of the book?', 'Would you like to hear a bit about the book?'],
        }

    def set_recommendation(self, book, summary) -> None:
        self.recommendation = book
        self.recommendation_summary = self.summarize(summary, min_length=20)[
            0]['summary_text']
        self.update_responses()

    def set_location(self, location) -> None:
        self.location = location
        self.update_responses()

    def get_country_name(self) -> str:
        if self.location == None:
            return None
        return pycountry.countries.get(alpha_2=self.location).name

    def get_answer(self, input, question) -> dict:
        print(f"input: {input}")
        print(f"question: {question}")
        if input == None or input == '':
            return None
        QA_input = {
            'question': question,
            'context': input
        }
        res = self.nlp(QA_input)
        print(f"res: {res}")
        if res['score'] < 0.001:
            return None
        return res

    def get_rating(self, text: str) -> int:
        res = self.sentiment(text)[0]
        if text != "" and res != None and res['label'] == 'POSITIVE':
            return int(res['score'] * 100)
        elif text != "" and res != None and res['label'] == 'NEGATIVE':
            return 100 - int(res['score'] * 100)
        return False

    def get_summary(self) -> str:
        return self.recommendation_summary

    def positive_response(self, text: str) -> bool:
        res = self.sentiment(text)[0]
        if text != "" and res != None and res['label'] == 'POSITIVE':
            return True
        return False

    def user_response(self, text) -> bool:

        questions = {
            'genre': 'What is the genre mentioned?',
            'author': 'Who is the author mentioned?',
            'book': 'What is the name of the book mentioned?',
            'rating': 'What is the rating mentioned?',
        }
        print(f'last state: {self.last_state}')
        print(f'state: {self.state}')
        if self.last_state not in questions.keys():
            return False
        print("User response: ", text)
        result = self.get_answer(text, questions[self.last_state])
        print(f"result: {result}")
        if result == None:
            return False

        if self.last_state == 'genre':
            self.genre = result['answer']
        elif self.last_state == 'author':
            author = self.rec.extract_first_author(
                self.rec.lookup_author(result['answer']))
            if author != None:
                self.author = author
            else:
                self.author = result['answer']
        elif self.last_state == 'book':
            book = self.rec.lookup_book_by_title(result['answer'])
            if book != None:
                self.book = book['title']
            else:
                self.book = result['answer']
        elif self.last_state == 'rating':
            self.rating = result['answer']

        self.update_responses()
        return True

    def _choose_option(self) -> str:
        options = ['genre', 'author', 'book', 'location']
        if self.genre != None:
            options.remove('genre')
            self.num_questions -= 1
        if self.author != None:
            options.remove('author')
            self.num_questions -= 1
        if self.book != None:
            options.remove('book')
            self.num_questions -= 1
        if len(options) == 1:
            self.num_questions = 0

        self.state = random.choice(options)
        self.num_questions -= 1

    def get_response(self, user_reaction: bool = False) -> str:
        self.last_state = self.state
        response = self.select_response()

        if self.state == 'greeting':
            self.state = 'mood'
        elif self.state == 'mood':
            if self.last_book != None:
                self.state = 'last_book'
            else:
                self._choose_option()
        elif self.state == 'last_book':
            self.state = 'rating'
        elif self.state == 'rating':
            self._choose_option()
        elif self.state in ['genre', 'author', 'book', 'location']:
            if self.num_questions >= 0:
                self._choose_option()
            else:
                self.state = 'recommendation'
        elif self.state == 'recommendation':
            self.state = 'summary'
            response += ", " + self.select_response()
        elif self.state == 'summary':
            self.last_state = self.state
            response = self.acknowledge()
            if user_reaction:
                response += ", " + self.recommendation_summary
            self.state = 'new_book'
            response += ", " + self.select_response()
        elif self.state == 'new_book':
            self.last_state = self.state
            response = self.acknowledge()
            if user_reaction:
                recom = self.rec.get_current_mood_recommendation(self.mood)
                self.set_recommendation(
                    recom['title'],
                    recom['description']
                )
                self.state = 'recommendation'
                response += ", " + self.select_response()
                self.state = 'summary'
                response += ", " + self.select_response()
            else:
                self.state = 'goodbye'

        return response

    def acknowledge(self) -> str:
        tmp = self.state
        self.state = 'confirm'
        res = self.select_response()
        self.state = tmp
        return res

    def select_response(self) -> str:
        return random.choice(self.responses[self.state])

    def load_knowledge(self, knowledge_json) -> None:
        if not os.path.exists(knowledge_json):
            self.save_knowledge(knowledge_json)

        with open(knowledge_json) as f:
            try:
                self.knowledge = json.load(f)
            except:
                self.save_knowledge(knowledge_json)
                self.knowledge = json.load(f)

        for id, knowledge in self.knowledge.items():

            self.genre = knowledge['genre']
            self.author = knowledge['author']
            self.book = knowledge['book']

            self.last_book = knowledge['recommendation']

    def save_knowledge(self, knowledge_json) -> None:
        # generate random alpha-numeric id for the knowledge
        id = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=10))
        new_knowledge = {
            'mood': self.mood,
            'genre': self.genre,
            'author': self.author,
            'book': self.book,
            'location': self.location,
            'last_book': self.last_book,
            'rating': self.rating,
            'recommendation': self.recommendation,
        }
        self.knowledge[id] = new_knowledge

        with open(knowledge_json, 'w') as f:
            json.dump(self.knowledge, f)

    def _get_previous(self, key) -> list:
        previous = []
        for id, knowledge in self.knowledge.items():
            previous.append(knowledge[key])
        return previous

    def get_last_books(self) -> list:
        books = self._get_previous('last_book')
        ratings = self._get_previous('rating')
        converted_ratings = []
        values = ["zero", "one", "two", "three", "four",
                  "five", "six", "seven", "eight", "nine", "ten"]
        for i in ratings:
            if i is None:
                i = "-1"
            if i in values:
                converted_ratings.append(values.index(i))
            else:
                try:
                    converted_ratings.append(int(i))
                except ValueError:
                    converted_ratings.append(-1)

        return [(books[i], converted_ratings[i]) for i in range(0, len(books))]

    def get_previous_authors(self) -> list:
        return self._get_previous('author')

    def get_previous_books(self) -> list:
        return self._get_previous('book')

    def get_previous_genres(self) -> list:
        return self._get_previous('genre')

    def get_previous_locations(self) -> list:
        return self._get_previous('location')

    def get_previous_recommendations(self) -> list:
        return self._get_previous('recommendation')

    def get_previous_ratings(self) -> list:
        return self._get_previous('rating')

    def get_previous_moods(self) -> list:
        return self._get_previous('mood')

    def __del__(self):
        self.save_knowledge('knowledge.json')
