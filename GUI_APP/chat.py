import random
import json
import sqlite3

import torch

from model import NeuralNet
from nltk_util import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
connection = sqlite3.connect('student_courses.db')
cursor = connection.cursor()

command1 = """CREATE TABLE IF NOT EXISTS 
student_courses(student_id INTEGER PRIMARY KEY, courses TEXT)"""

cursor.execute(command1)

cursor.execute(
    "INSERT INTO student_courses VALUES (17090,'Course A, Course B')")
cursor.execute(
    "INSERT INTO student_courses VALUES (17091,'Course C, Course D')")
cursor.execute(
    "INSERT INTO student_courses VALUES (17092,'Course E, Course F')")

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
print("Let's chat! (type 'quit' to exit)")
state = None
previous_question = None


def get_response(msg, state=None, previous_question=None, student_id=0):
    bot_name = "Bot"
    #print("Let's chat! (type 'quit' to exit)")
    # state = None
    # previous_question = None

    # student_id = 0

    sentence = msg
    if state == "faculty":
        sentence += " faculty list"
        print(sentence)
    elif state == "course_registered":
        student_id = int(sentence)
        sentence = f'12341234 {previous_question}'
        print(sentence)
    elif state == "course_details" or state == "next_semester_course_registered" or state == "waiting_list":
        sentence += f' {previous_question}'
        print(sentence)
    else:
        state = None
        previous_question = None

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    # print(tag)
    if tag == "department_faculty":
        state = "faculty"
    elif tag == "course_registered":
        state = "course_registered"
        previous_question = " ".join(sentence)
    elif tag == "course_details":
        state = "course_details"
        previous_question = " ".join(sentence)
    elif tag == "next_semester_course_registered":
        state = "next_semester_course_registered"
        previous_question = " ".join(sentence)
    elif tag == "waiting_list":
        state = "waiting_list"
        previous_question = " ".join(sentence)
    else:
        state = None
        previous_question = None

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                current_output = random.choice(intent['responses'])
                if current_output == "Your courses are: Course1, Course2 and Course3":
                    cursor.execute(
                        f'SELECT courses FROM student_courses WHERE student_id={student_id}')
                    results = cursor.fetchall()
                    if len(results) == 0:
                        return(f"{bot_name}: Invalid Id", state, previous_question, student_id)
                    else:
                        return(f"{bot_name}: {results[0][0]}", state, previous_question, student_id)
                else:
                    return(f"{bot_name}: {random.choice(intent['responses'])}", state, previous_question, student_id)
    else:
        return(f"{bot_name}: I do not understand...", state, previous_question, student_id)
