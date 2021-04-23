import numpy as np

genre_embedding = np.load('C:/Users/cody1/Downloads/py_tut/rankers/genre_embeddings.npy')

word1 = []
for i in genre_embedding:
    word1.append(i[0])
    
word2 = []
for i in genre_embedding:
    word2.append(i[1]) 
    
word2 = np.array(word2)
    
word1 = np.array(word1)

genre_embedding = np.concatenate((word1,word2), axis=1)

np.save('C:/Users/cody1/Downloads/py_tut/rankers/genre_embeddings.npy', genre_embedding)
