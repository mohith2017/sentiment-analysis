import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_imdb_data(data_dir):
    data = {'review': [], 'sentiment': []}
    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, sentiment)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                data['review'].append(f.read())
                data['sentiment'].append(0 if sentiment == 'neg' else 1)
    return pd.DataFrame(data)

train_data = load_imdb_data('./aclImdb/train')

test_data = load_imdb_data('./aclImdb/test')

all_data = pd.concat([train_data, test_data], ignore_index=True)
all_data.to_csv('data/IMDB Dataset.csv')