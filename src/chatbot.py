import json
from pathlib import Path
import random
import string
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"


class chatbot:

    def __init__(self, knowledge_json=None) -> None:
        self.state = 'greeting'
        self.last_state = None

        self.nlp = pipeline('question-answering',
                            model=model_name, tokenizer=model_name)

        self.mood = None
        self.genre = None
        self.author = None
        self.book = None
        self.location = None
        self.last_book = None
        self.recommendation = None

        self.responses = {}

        self.knowledge = {}
        if knowledge_json:
            self.load_knowledge(knowledge_json)

        self.update_responses()

    def update_responses(self) -> None:
        self.responses = {
            'greeting': ['Hello', 'Hi', 'Welcome', 'Good day', 'Hey'],
            'confirm': ['OK', 'alright', 'I see', 'I get it'],
            'mood': ['How are you?', 'How are you doing?', 'How are you feeling?', 'How are you today?'],
            'genre': ['What genre do you like?', 'What genre do you prefer?', 'What genre do you enjoy?', 'What genre do you read?'],
            'author': ['Who is your favorite author?', 'Which author do you like?', 'Name an author that you like'],
            'book': ['What is your favorite book?', 'Which book have you liked?', 'Name a book that you like'],
            'location': [f'Would you like to read an author from {self.location}?', f'Would you like to read a book from {self.location}?'],
            'last_book': [f'Did you like the {self.last_book}?', f'Did you enjoy {self.last_book}?', f'Have you finished {self.last_book}?'],
            'recommendation': [f'I recommend you read {self.recommendation}', f'I think you should read {self.recommendation}', f'I suggest you look into {self.recommendation}', f'I think you would like {self.recommendation}'],
            'goodbye': ['Goodbye', 'See you later', 'Bye', 'Have a nice day', 'Have a good day']
        }

    def set_recommendation(self, book) -> None:
        self.recommendation = book
        self.update_responses()

    def set_location(self, location) -> None:
        self.location = location
        self.update_responses()

    def get_answer(self, input, question) -> dict:
        if input == None or input == '':
            return None
        QA_input = {
            'question': question,
            'context': input
        }
        res = self.nlp(QA_input)
        print(res)
        if res['score'] < 0.5:
            return None
        return res

    def user_response(self, text) -> bool:
        questions = {'genre': 'What is the genre mentioned?',
                     'author': 'Who is the book author mentioned?', 'book': 'What is the book mentioned?'}
        if self.last_state not in questions.keys():
            return False

        result = self.get_answer(text, questions[self.last_state])
        if result == None:
            return False

        if self.last_state == 'genre':
            self.genre = result['answer']
        elif self.last_state == 'author':
            self.author = result['answer']
        elif self.last_state == 'book':
            self.book = result['answer']

        self.update_responses()
        return True

    def _choose_option(self) -> str:
        options = ['genre', 'author', 'book', 'location']
        if self.genre != None:
            options.remove('genre')
        if self.author != None:
            options.remove('author')
        if self.book != None:
            options.remove('book')

        self.state = random.choice(options)

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
            self._choose_option()
        elif self.state in ['genre', 'author', 'book', 'location']:
            self.state = 'recommendation'
        elif self.state == 'recommendation':
            self.state = 'goodbye'

        return response

    def select_response(self) -> str:
        return random.choice(self.responses[self.state])

    def load_knowledge(self, knowledge_json) -> None:
        with open(knowledge_json) as f:
            self.knowledge = json.load(f)

        for id, knowledge in self.knowledge.items():
            if self.genre == None:
                self.genre = knowledge['genre']
            if self.author == None:
                self.author = knowledge['author']
            if self.book == None:
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
            'recommendation': self.recommendation,
        }
        self.knowledge[id] = new_knowledge

        with open(knowledge_json, 'w') as f:
            json.dump(self.knowledge, f)

    def __del__(self):
        self.save_knowledge('knowledge.json')
