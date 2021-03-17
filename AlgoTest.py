import numpy as np
import pandas as pd

spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_campare = spotify[['name','artists']]


spotify_slim = pd.read_csv('C:\\Users\\cody1\\Downloads\\py tut\\rankers\\spotslim.csv', engine='python', encoding='latin-1')
spotify_slim.set_index('name', inplace=True)

useable_songs = spotify_slim.iloc[:,2:-1]

codysrankSM = spotify_slim[spotify_slim['rank']>0]
codysrankSM = codysrankSM.iloc[:,::]

#values of the finding_songs()
m_low = .9
m_high = 1.1
'''Set values to 0 if not used'''
#values for more customizable finding_song_custom()
c0 = [.70, 1.1] #dance
c1 = [.95, 1.15] #energy
c2 = 0 #key can also be a bool
use_loudness = False # if using loudness set to true
c3 = [.90, 1.1] #loudness
c4 = True #mode (mode is a binary value 1 or 0)
c5 = [.8, 1.2] #acoustic may work better as a bool check to see if its above or below a certain num
c6 = [.90, 1.1] #instrumentalness
c7 = [.90, 1.1] #temp
c8 = [.80, 1.2] #liveness (whether or not is was performed live)
c9 = [.80, 1.2] #valence (how positive or negitive a track sounds)
c10 = [.99, 1.01] #year

score = 8 # how many criterias they must meet. Make sure score doesnt over reach number of features
'''if score is too high you wont get good results, if any'''


song_in_question = 508917 #index

'''the left & right concat is optional, just there incase you want to edit the input columns'''
left = useable_songs.iloc[:, 0:3]
right = useable_songs.iloc[:, 3:]
songs = pd.concat([left,right], axis = 1)
songs = songs.values.tolist()


def finding_songs(songs, song_to_compare, m_low, m_high, score):
    songlist = []
    for song in songs:
        _songpoints = 0
        for a,b in zip(song, song_to_compare):
            if a >= b*m_low and a <= b*m_high :
                _songpoints +=1
                #print(_songpoints)
            if _songpoints == score:
                songlist.append(song)
                #print (song)
                #print (song_to_compare)
                
            
    return songlist


def finding_song_custom(songs, song_to_compare, c0 = 0., c1 = 0., c2 = 0., use_loudness = False, c3 = 0., c4 = False, c5 = 0., c6 = 0., c7 = 0., c8 = 0., c9 = 0., c10 = 0., score = 7):
    songlist = []
    for song in songs:
        _songpoints = 0
        if c0 != 0: 
            if song[0] >= song_to_compare[0]*c0[0] and song[0] <= song_to_compare[0]*c0[1]:
                _songpoints +=1
        if c1 != 0: 
            if song[1] >= song_to_compare[1]*c1[0] and song[1] <= song_to_compare[1]*c1[1]:
                _songpoints +=1
        if c2 != 0: 
            if song[2] >= song_to_compare[2]*c2[0] and song[2] <= song_to_compare[2]*c2[1]:
                _songpoints +=1
        if use_loudness != False:
            if c3 != 0: 
                if song[3] >= song_to_compare[3]*c3[0] and song[3] <= song_to_compare[3]*c3[1]:
                    _songpoints +=1
        if c4 != False: 
            if song[4] == song_to_compare[4]:
                _songpoints +=1
        if c5 != 0: 
            if song[5] >= song_to_compare[5]*c5[0] and song[5] <= song_to_compare[5]*c5[1]:
                _songpoints +=1
        if c6 != 0: 
            if song[6] >= song_to_compare[6]*c6[0] and song[6] <= song_to_compare[6]*c6[1]:
                _songpoints +=1
        if c7 != 0: 
            if song[7] >= song_to_compare[7]*c7[0] and song[7] <= song_to_compare[7]*c7[1]:
                _songpoints +=1
        if c8 != 0: 
            if song[8] >= song_to_compare[8]*c8[0] and song[8] <= song_to_compare[8]*c8[1]:
                _songpoints +=1
        if c9 != 0: 
            if song[9] >= song_to_compare[9]*c9[0] and song[9] <= song_to_compare[9]*c9[1]:
                _songpoints +=1
        if c10 != 0: 
            if song[10] >= song_to_compare[10]*c10[0] and song[10] <= song_to_compare[10]*c10[1]:
                _songpoints +=1
        if _songpoints >= score:
                songlist.append(song)
                print (song)
                #print (song_to_compare)
            
    return songlist
            
#spotify_campare.iloc[291462, :]
#spotify_campare.iloc[1059423, :]


def re_grabbing_keys(songs, list_songs):
    song_with_key = {}
    key = 0
    where_at = 0
    for song in songs:
        key +=1
        for csong in list_songs:
            _index = 0
            for a,b in zip(song,csong):
                if a == b:
                    _index +=1
                
                if _index == 11:
                    song_with_key[key-1] = song
                    where_at +=1
                    print(where_at)
    return song_with_key

#simple algo
list_songs = finding_songs(songs, songs[song_in_question], m_low, m_high, score)
#custom algo
list_songs = finding_song_custom(songs, songs[song_in_question], c0, c1, c2, use_loudness, c3, c4, c5, c6, c7, c8, c9, c10, score)
                
song_keys = re_grabbing_keys(songs, list_songs)

master_key = list(song_keys.keys())

rec_songs = spotify_campare.loc[master_key]
                
            

'''the full list and adding 1 after every loop to and connecting the song on that lopp with the key'''
            

            

                



