import numpy as np
import pandas as pd
import tensorflow as tf

#grab the file here 
'''https://www.kaggle.com/rodolfofigueroa/spotify-12m-songs'''

spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_campare = spotify[['name','artists']]


spotify_slim = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\spotslim.csv', engine='python', encoding='latin-1')
spotify_slim.set_index('name', inplace=True)

# genre_token = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\genre_token.csv')
# genre_token = genre_token.iloc[:,1:]

genre_embedding = np.load('C:/Users/cody1/Downloads/py_tut/rankers/genre_embeddings.npy')


#whereimat= spotify_slim[spotify_slim['rank']>0]
codysrank = spotify_slim[['danceability', 'energy', 'mode', 'acousticness', 'year', 'rank' ]]


codysrank = np.concatenate((genre_embedding, np.array(codysrank)), axis=1)

codysrank = pd.DataFrame(codysrank)


#codysrank.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\codysrank.csv')
codysrankSM = codysrank[codysrank[9]>0]

ranked = codysrankSM.iloc[:,-1]
SPfeatures = codysrankSM.iloc[:,:-1]
SPfeatures = np.array(SPfeatures)


#if activated, must change activation loss to binary and activatin to sigmoid
ranked[ranked <=3 ] = 0
ranked[ranked > 3] = 1

ranked = ranked.values.tolist()
ranked_ = []
for rank in ranked:
    if rank == 1.0:
        ranked_.append([0,1])
    else:
        ranked_.append([1,0])

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# #currently unused only needed if you have more than 2 catagories
# ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(),[1])], remainder='passthrough')
# ranked = ct.fit_transform(ranked)

# ranked= pd.DataFrame(ranked)
# ranked= ranked.iloc[:,:-1]
ranked = np.array(ranked_)




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(SPfeatures, ranked, test_size = 0.2)

'''
X_train = SPfeatures
y_train = ranked
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
adam = tf.keras.optimizers.Adam(
    learning_rate=0.05,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=True,
    name="adam"
)

ann.add(tf.keras.layers.Dense(units=20, activation='softplus'))
#ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=9, activation='softplus'))
#ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))

ann.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
    #softmax
    
ann.compile(optimizer=adam , loss= 'binary_crossentropy')
    #categorical_crossentropy
    
ann.fit(X_train, y_train, batch_size=16, epochs = 400, verbose=1)
    
#ann = tf.keras.models.load_model('C:/Users/cody1/Downloads/py_tut/rankers/Saved/acc71_softplus_lr15.h5')
    #must be 2d array
accuracy_test = ann.predict(X_test)
#clipping to avoid log(0)
clip_pred = np.clip(accuracy_test, 1e-7, 1 - 1e-7)
#calculating loss
loss = np.mean(-np.log(np.sum(clip_pred * y_test, axis=1)))
#calculating accuracy
accuracy = np.mean(np.argmax(accuracy_test, axis=1) == np.argmax(y_test, axis=1))

print("Loss: ", loss)
print("accuracy: ", accuracy)

ann.reset_states()

tf.keras.models.save_model(model= ann, filepath='C:/Users/cody1/Downloads/py_tut/rankers/Saved/acc71_softplus_lr15.h5')

#applying to a bigger data set
codysrank = codysrank.iloc[:,:-1]
codysrank = np.array(codysrank)
testedpred= ann.predict(sc.transform(codysrank))

pop = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_pop.csv', engine='python', encoding='latin-1')
pop = np.array(pop.iloc[:,1:])

testedpred = np.concatenate((testedpred, pop), axis=1)
#accuracytest= pd.DataFrame(accuracytest)
#conveting array to dataframe
testedpred= pd.DataFrame(testedpred)
#combining a easier to read dataframe
results= testedpred.iloc[:,-2:]
codys = pd.concat([spotify_campare,results],axis=1)
#grabbing only songs it is confidient i will like
codyslim= codys[codys.iloc[:,-1]>40]
codyslim= codyslim[codyslim.iloc[:,-2]>.7]

testedpred.to_csv('C:/Users/cody1/Downloads/py_tut/rankers/Saved/w_embedding_song.csv')
codyslim.to_csv('C:/Users/cody1/Downloads/py_tut/rankers/Saved/norrow_song.csv')