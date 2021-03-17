import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#import dataset
spotify = pd.read_csv('tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_campare = spotify[['name','artists']]
spotify_campare['rank']= np.zeros(1204025)
#spotify_campare.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotcomp.csv')

#spotify_slim= spotify[['name','id','danceability','energy','key','loudness','mode','acousticness','instrumentalness','tempo','liveness','valence','year']]

#spotify_slim['rank']= np.zeros(1204025)
spotify_slim = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotslim.csv', engine='python', encoding='latin-1')
#spotify_slim['rank'][787631]= 4
#spotify_slim.iloc[1165102]
spotify_slim.set_index('name', inplace=True)
spotify_codes = spotify_slim.iloc[:,2:]
spotify_slim.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotslim.csv')
spotify_codes.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spot_slimed.csv')
'''
get rid of id column
'''
#remembering where im at
whereimat= spotify_slim[spotify_slim['rank']>0]

#time to move on
codysrank = whereimat.iloc[:,2:]
codysrank.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\codysrank.csv')
codysrank = codysrank[spotify_slim['rank']>0]


training_set = codysrank.iloc[:90,:]
test_set= codysrank.iloc[90:,:]
training_set= np.array(training_set)
test_set= np.array(test_set)

training_set = training_set.tolist()
test_set = test_set.tolist()
#convert to torch tensors

training_set= torch.FloatTensor(training_set)
test_set= torch.FloatTensor(test_set)

#creating architec of nn
#inheriting Module from torch
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(12, 50) #first layer in the network features, nodes
        self.fc2 = nn.Linear(50, 20) #second layer nuerons from first layer, nodes
        self.fc3 = nn.Linear(20, 50) # its like a perimid 
        self.fc4 = nn.Linear(50, 12)
        self.activation = nn.Sigmoid()
    def forward(self, x):           #input vector foward propigating, encoding to decoding
        x= self.activation(self.fc1(x))
        x= self.activation(self.fc2(x))
        x= self.activation(self.fc3(x))
        x= self.fc4(x)      #since we are decoding we dont want to apply the activation func
        return x
    
sae = SAE()

criterion = nn.MSELoss()

optimizer = optim.RMSprop(sae.parameters(), lr= 0.01, weight_decay=0.5 ) #need parameters

#training
#define number of epoch
num_epoch = 50
for epoch  in range(1,num_epoch+1): #the +one is because it goes up to but not including. so plus 1 adds the last one
    train_loss = 0.
    for rank in range(len(training_set)): #everything happening in 1 epoch
        input = Variable(training_set[rank]).unsqueeze(0) #adding a new dimesion
        target = input.clone()
        
        output = sae(input)
        target.require_grad = False
        output[target ==0] = 0
        loss = criterion(output, target)
        mean_corrector = len(training_set)/float(torch.sum(target>0) + 1e-10) #want to prevent non 0 numbers
        loss.backward() #desides directon the weight with either increase or decrease
        train_loss += np.sqrt(loss.data*mean_corrector)
            
        optimizer.step() #deside the amount the weight will increase or decrease
    print('epoch: '+str(epoch)+' loss: '+str(train_loss))
    
    
    
#testing the data
test_loss = 0.
for rank in range(len(test_set)): #everything happening in 1 epoch
    input = Variable(training_set[rank]).unsqueeze(0) #adding a new dimesion
    target = Variable(test_set[rank]).unsqueeze(0)
        
    output = sae(input)
    target.require_grad = False
   
    loss = criterion(output, target)
    mean_corrector = len(test_set)/float(torch.sum(target>0) + 1e-10) #want to prevent non 0 numbers
    test_loss += np.sqrt(loss.data*mean_corrector)
    
print('loss: '+str(test_loss))


resultss= output.tolist()
resultw= np.array(resultss)    
df= pd.DataFrame(resultw, columns=['danceability','energy','key','loudness','mode','acousticness','instrumentalness','tempo','liveness','valence','year','rank'])
