import random
import json
import numpy as np

import torch

from model import NeuralNet
from nltk_util import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('testing_set.json', 'r') as f:
    test_data = json.load(f)

with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

total_cases_till_now = 0
correct_cases = 0
epoch = 0

for item in test_data:
    total_cases_till_now += 1
    sentence = item["q"]
    answer = item["a"]

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    epoch += 1
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                current_output = random.choice(intent['responses'])
                if current_output == answer:
                    correct_cases += 1

    print(
        f'Epoch [{epoch}/{len(test_data)}], Accuracy: {correct_cases/total_cases_till_now}')
