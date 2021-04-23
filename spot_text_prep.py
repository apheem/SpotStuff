'''spotify text preprocessing'''

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

genres = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_genres.csv')
genres = np.array(genres.iloc[:,1:])
genres = genres.tolist()

pop = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_pop.csv')
pop = np.array(pop.iloc[:,1:])

for genre in range(len(genres)):
    for i in range(len(genres[genre])):
        if isinstance(genres[genre][i], float):
            genres[genre][i] = 'None'
        



tokenizer = Tokenizer(num_words = 20000, split= ',  ')
tokenizer.fit_on_texts(genres)
sequence = tokenizer.texts_to_sequences(genres)

genres_token = np.array(sequence)
# genre_token_save = pd.DataFrame(genres_token)
# genre_token_save.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\genre_token.csv')
        
song_length = genres_token.shape[0]
T = genres_token.shape[1]      
embedding_vector = 2
hidden_state_dim = 15

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

genre_i = Input(shape= (T,))
g_embedding = Embedding(song_length, embedding_vector, input_length=T)(genre_i)

model = Model(genre_i, g_embedding)
print(model.summary())

model.compile(loss='mse', optimizer='rmsprop')

outp = model.predict(genres_token)

genre_embedding = np.array(outp)
np.save('C:/Users/cody1/Downloads/py_tut/rankers/genre_embeddings.npy', genre_embedding)

headb = genre_embedding[:5]
