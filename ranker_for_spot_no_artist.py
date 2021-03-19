import numpy as np
import pandas as pd
import keras
import tensorflow as tf

spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_campare = spotify[['name','artists']]


spotify_slim = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\spotslim.csv', engine='python', encoding='latin-1')
spotify_slim.set_index('name', inplace=True)

#whereimat= spotify_slim[spotify_slim['rank']>0]
codysrank = spotify_slim.iloc[:,2:]
#codysrank.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\codysrank.csv')
codysrankSM = codysrank[codysrank['rank']>0]

ranked = codysrankSM.iloc[:,-2:]
SPfeatures = codysrankSM.iloc[:,:-1]
SPfeatures = np.array(SPfeatures)


#if activated, must change activation loss to binary and activatin to sigmoid
ranked[ranked <=3 ] = 0
ranked[ranked > 3] = 1

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(),[1])], remainder='passthrough')
ranked = ct.fit_transform(ranked)

ranked= pd.DataFrame(ranked)
ranked= ranked.iloc[:,:-1]
ranked = np.array(ranked)




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(SPfeatures, ranked, test_size = 0.1)

'''
X_train = SPfeatures
y_train = ranked
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ann = tf.keras.models.Sequential()
lowest_loss = 99.

best_first = 0
best_sec = 0
best_third = 0
best_fourth = 0

for iteration in range(74):
    first_layer = np.array(range(1,26,1))
    sec_layer = np.array(range(1,26,1))
    third_layer = 0
    
    print(iteration)
    if iteration <= 24:        
        ann.add(tf.keras.layers.Dense(units=first_layer[iteration], activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer[iteration], activation='relu'))
        
    elif 24 < iteration<= 49:
        third_layer = iteration-24
        ann.add(tf.keras.layers.Dense(units=first_layer[iteration-25], activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer[iteration-25], activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
    
    elif iteration > 49:
        third_layer = iteration-48
        fourth_layer = iteration-48
        ann.add(tf.keras.layers.Dense(units=first_layer[iteration-49], activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer[iteration-49], activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=fourth_layer, activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
    #softmax
    #try linear 
    
    ann.compile(optimizer='adam' , loss= 'binary_crossentropy', validation_data=(X_train,y_train),metric= 'accuracy')
    #categorical_crossentropy
    ann.fit(X_train, y_train, batch_size=1, epochs = 200, verbose=0)
    

    #must be 2d array
    accuracy_test = ann.predict(X_test)
    if iteration <= 24:
        if np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1))) < lowest_loss:
            best_first = first_layer[iteration]
            best_sec = sec_layer[iteration]
            lowest_loss = np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1)))
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('No Third Layer')
            print('No Fourth Layer')

    if iteration > 24 & iteration <= 48:
        if np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1))) < lowest_loss:
            best_first = first_layer[iteration-25]
            best_sec = sec_layer[iteration-25]
            best_third = third_layer
            lowest_loss = np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1)))
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('Third Layer:', best_third)
            print('No Fourth Layer')
            
    if iteration > 48 :
        if np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1))) < lowest_loss:
            best_first = first_layer[iteration-25]
            best_sec = sec_layer[iteration-25]
            best_third = third_layer
            best_fourth = fourth_layer
            lowest_loss = np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1)))
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('Third Layer:', best_third)
            print('Best Fourth:', best_fourth)
        
    ann.reset_states() 
    ann.reset_metrics()
    
'''random int'''
'''rand int'''
'''random int'''
for iteration in range(160000):
    first_layer = np.random.randint(7,26)
    sec_layer = np.random.randint(7,26)
    third_layer = np.random.randint(7,26)
    fourth_layer = np.random.randint(7,26)
    
    print(iteration)
    if iteration <= 399:        
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        
    elif 399 < iteration <= 8000:
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
    
    elif iteration > 8000:
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=fourth_layer, activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
    #softmax
    #try linear 
    
    ann.compile(optimizer='adam' , loss= 'binary_crossentropy', validation_data=(X_train,y_train),metric= 'accuracy')
    #categorical_crossentropy
    ann.fit(X_train, y_train, batch_size=1, epochs = 300, verbose=0)
    

    #must be 2d array
    accuracy_test = ann.predict(X_test)
    clip_pred = np.clip(accuracy_test, 1e-7, 1e-7)
    loss = np.mean(-np.log(np.sum(clip_pred * y_test, axis=1)))
    print('Loss: ', loss)
    if iteration <= 399:
        if loss < lowest_loss:
            best_first = first_layer
            best_sec = sec_layer
            lowest_loss = loss
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('No Third Layer')
            print('No Fourth Layer')

    if 399 < iteration <= 8000:
        if loss < lowest_loss:
            best_first = first_layer
            best_sec = sec_layer
            best_third = third_layer
            lowest_loss = loss
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('Third Layer:', best_third)
            print('No Fourth Layer')
            
    if iteration > 8000:
        if loss < lowest_loss:
            best_first = first_layer
            best_sec = sec_layer
            best_third = third_layer
            best_fourth = fourth_layer
            lowest_loss = loss
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('Third Layer:', best_third)
            print('Best Fourth:', best_fourth)
        
    ann.reset_states() 
    ann.reset_metrics()

codysrank = codysrank.iloc[:,:-1]
codysrank = np.array(codysrank)
testedpred= ann.predict(sc.transform(codysrank))

#accuracytest= pd.DataFrame(accuracytest)
testedpred= pd.DataFrame(testedpred)

results= testedpred.iloc[:,-1]

codys = pd.concat([spotify_campare,results],axis=1)

codyslim= codys[codys.iloc[:,-1]>0.85]

codyslim.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\codysrecommend22k.csv')
