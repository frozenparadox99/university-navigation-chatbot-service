import numpy as np
import random
import json

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_util import bag_of_words, tokenize, stem
from model import NeuralNet

with open("intent.json", "r") as f:
    intents = json.load(f)

plt.style.use("seaborn")

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]

    tags.append(tag)
    for pattern in intent["patterns"]:

        w = tokenize(pattern)

        all_words.extend(w)

        xy.append((w, tag))


ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)


X_train = []
y_train = []
for (pattern_sentence, tag) in xy:

    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

learing_rate_wise_loss_list = {}

different_learning_rates = [0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

for curr_rate in different_learning_rates:
    print(f'\nLearning Rate: {curr_rate}')

    num_epochs = 1000
    batch_size = 8
    learning_rate = curr_rate
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    print(input_size, output_size)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    #
    dataset = ChatDataset()
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            loss_list.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print(f"final loss: {loss.item():.4f}")

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }

    FILE = "data.pth"
    #torch.save(data, FILE)

    learing_rate_wise_loss_list[curr_rate] = loss_list

    # plt.plot(loss_list)
    # plt.title("Epoch vs Loss (MLP)")
    # plt.xlabel("Epochs")
    # plt.ylabel("Cross Entropy Loss")
    # plt.show()

with open("result_learning_rate.json", "w") as outfile:
    json.dump(learing_rate_wise_loss_list, outfile)
