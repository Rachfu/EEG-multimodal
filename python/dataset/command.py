import pandas as pd

df = pd.read_csv("test_EEG.csv")
df = df.iloc[:,1:]
df.to_csv("test_EEG.csv",index=False)
print(df.shape)
