import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_json('result_hidden_layer.json')

print(df.to_string())

plt.figure()
plt.title("Epoch vs Loss (MLP)")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
df[[14, 15, 16, 17]].plot()
plt.show()
