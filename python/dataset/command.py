import pandas as pd
import pickle
with open("../embedding/EEG/txt/bert_bert_base_uncased/test.pickle", 'rb') as f:
    train_clip_feature_now = pickle.load(f)
print(train_clip_feature_now)
with open("../../feature/EEG/test_bert.pickle", 'rb') as f:
    train_clip_feature_before = pickle.load(f)
print(train_clip_feature_before)

print(train_clip_feature_now == train_clip_feature_now)
    