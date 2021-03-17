import numpy as np
import pandas as pd
import keras
import tensorflow as tf

spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_campare = spotify[['name','artists']]

spotify_slim = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotslim.csv', engine='python', encoding='latin-1')
spotify_slim.set_index('name', inplace=True)

#whereimat= spotify_slim[spotify_slim['rank']>0]
codysrank = spotify_slim.iloc[:,2:]
#codysrank.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\codysrank.csv')
codysrankSM = codysrank[codysrank['rank']>0]

ranked = codysrankSM.iloc[:,-2:]
SPfeatures = codysrankSM.iloc[:,:-1]
ranked = np.array(ranked)
SPfeatures = np.array(SPfeatures)
'''
ranked[ranked <=3 ] = 0
ranked[ranked > 3] = 1
'''



'''
for i in ranked:
    if i > 3 :
        ranked[i] = 1.0
    else:
        ranked[i] = 0.0
 '''       
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(),[1])], remainder='passthrough')
ranked = ct.fit_transform(ranked)

ranked= pd.DataFrame(ranked)
ranked= ranked.iloc[:,:-1]
ranked = np.array(ranked)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(SPfeatures, ranked, test_size = 0.2)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=20, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
#ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=5, activation='softmax'))
#try linear 

ann.compile(optimizer='adam' , loss= 'categorical_crossentropy', validation_data=(X_train,y_train),metric= 'accuracy')

ann.fit(X_train, y_train, batch_size=16, epochs = 1000)


codysrank = codysrank.iloc[:,:-1]
codysrank = np.array(codysrank)
#must be 2d array
testedpred= ann.predict(sc.transform(X_test))

testedpred= pd.DataFrame(testedpred)

results= testedpred.iloc[:,-1]

codys = pd.concat([spotify_campare,results],axis=1)

codyslim= codys[codys.iloc[:,-1]>0.980]
