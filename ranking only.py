import numpy as np
import pandas as pd


#full data frame
spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_compare = spotify[['name','artists']]

#data from with ranks
spotify_slim = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotslim.csv', engine='python', encoding='latin-1')
spotify_slim.set_index('name', inplace=True)

#data frame with predictions
pred_ranked = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\codysrecommend.csv', engine='python', encoding='latin-1')

'''
spotify_compare1 = spotify_compare.iloc[:300000,:]
spotify_compare2 = spotify_compare.iloc[len(spotify_compare1):600000,:]
spotify_compare3 = spotify_compare.iloc[len(spotify_compare1)+len(spotify_compare2):900000,:]
spotify_compare4 = spotify_compare.iloc[len(spotify_compare1)+len(spotify_compare2)+len(spotify_compare3):,:]
'''
'''
spotify_compare1.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spot_comp1.csv')
spotify_compare2.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spot_comp2.csv')
spotify_compare3.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spot_comp3.csv')
spotify_compare4.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spot_comp4.csv')
'''

#to rank
spotify_slim['rank'][5583]= 2

#to check where you are
whereimat= spotify_slim[spotify_slim['rank']>0]

#save
spotify_slim.to_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotslim.csv')
