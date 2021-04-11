import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

with open('intent.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)

        all_words.extend(w)
        tags.append(intent['tag'])
        xy.append((w, intent['tag']))


words = [lemmatizer.lemmatize(w.lower())
         for w in all_words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(tags)))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
