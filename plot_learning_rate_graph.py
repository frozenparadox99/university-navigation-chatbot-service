import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_json('result_learning_rate.json')
df.columns = [0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

print(df.to_string())

plt.figure()
df.iloc[:, 4:].plot()

plt.title("Epoch vs Loss (MLP)")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")

plt.show()
