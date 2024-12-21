import pandas as pd

df = pd.read_csv("train_EEG.csv")
print(df.loc[1].tolist())
print(" ".join([str(i) for i in df.loc[1].tolist()]))
