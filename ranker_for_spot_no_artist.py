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

for i in range(1,161):
    print(i)
ann = tf.keras.models.Sequential()
lowest_loss = 99.

best_first = 0
best_sec = 0
best_third = 0
best_fourth = 0
best_fifth = 0
best_sixth = 0

for iteration in range(10,161):
    first_layer = iteration
    sec_layer = iteration
    third_layer = 0
    fourth_layer = 0
    fifth_layer = 0
    sixth_layer = 0
    
    print(iteration)
    if iteration <= 30: 
        first_layer = iteration
        sec_layer = iteration
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        
    elif 30 < iteration <= 59:
        first_layer = iteration-30
        sec_layer = iteration-30
        third_layer = iteration-30
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
    
    elif 59 < iteration <= 89:
        first_layer = iteration-59
        sec_layer = iteration-59
        third_layer = iteration-59
        fourth_layer = iteration-59
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=fourth_layer, activation='relu'))
    
    elif  89 < iteration <= 119:
        first_layer = iteration-89
        sec_layer = iteration-89
        third_layer = iteration-89
        fourth_layer = iteration-89
        fifth_layer = iteration-89
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=fourth_layer, activation='relu')) 
        ann.add(tf.keras.layers.Dense(units=fifth_layer, activation='relu')) 
        
    elif  119 < iteration <= 160:
        first_layer = iteration-119
        sec_layer = iteration-119
        third_layer = iteration-119
        fourth_layer = iteration-119
        fifth_layer = iteration-119
        sixth_layer = iteration-119
        ann.add(tf.keras.layers.Dense(units=first_layer, activation='relu'))
        #ann.add(tf.keras.layers.Dropout(0.2))
        ann.add(tf.keras.layers.Dense(units=sec_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=third_layer, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=fourth_layer, activation='relu')) 
        ann.add(tf.keras.layers.Dense(units=fifth_layer, activation='relu')) 
        ann.add(tf.keras.layers.Dense(units=sixth_layer, activation='relu')) 
        
    ann.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
    #softmax
    #try linear 
    
    ann.compile(optimizer='adam' , loss= 'binary_crossentropy', validation_data=(X_train,y_train),metric= 'accuracy')
    #categorical_crossentropy
    ann.fit(X_train, y_train, batch_size=1, epochs = 100, verbose=0)
    

    #must be 2d array
    accuracy_test = ann.predict(X_test)
    loss = np.mean(-np.log(np.sum(accuracy_test * y_test, axis=1)))   
    accuracy = np.mean(np.argmax(accuracy_test, axis=1) == np.argmax(y_test, axis=1))
    
    print('Accuracy: ', accuracy)
    
    if iteration <= 30:
        if loss < lowest_loss:
            best_first = first_layer
            best_sec = sec_layer
            lowest_loss = loss
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('No Third Layer')
            print('No Fourth Layer')

    if 30 < iteration <= 59:
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
            
    if 59 < iteration <= 89:
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
            
    if 89 < iteration <= 119:
        if loss < lowest_loss:
            best_first = first_layer
            best_sec = sec_layer
            best_third = third_layer
            best_fourth = fourth_layer
            best_fifth = fifth_layer
            lowest_loss = loss
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('Third Layer:', best_third)
            print('Best Fourth:', best_fourth)
            print('Best fifth: ', best_fifth)
            
    if iteration > 119:
        if loss < lowest_loss:
            best_first = first_layer
            best_sec = sec_layer
            best_third = third_layer
            best_fourth = fourth_layer
            best_fifth = fifth_layer
            best_sixth = sixth_layer
            lowest_loss = loss
            print('New best: ', lowest_loss)
            print('First Layer: ', best_first)
            print('Sec Layer: ', best_sec)
            print('Third Layer:', best_third)
            print('Best Fourth:', best_fourth)
            print('Best fifth: ', best_fifth)
            print('Best Sixth: ', best_sixth)
        
            
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
    ann.fit(X_train, y_train, batch_size=1, epochs = 250, verbose=0)
    

    #must be 2d array
    accuracy_test = ann.predict(X_test)
    clip_pred = np.clip(accuracy_test, 1e-7, 1 - 1e-7)
    loss =np.mean(-np.log(np.sum(clip_pred * y_test, axis=1)))
    
    
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
