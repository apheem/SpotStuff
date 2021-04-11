import numpy as np
import pandas as pd
import re
#my client id on my computer
clientids = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\clientids.csv')

spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
#refer to rank only on git to rank songs or rank them in a csv editor
spotify_campare = spotify[['name','artists']]
#loading saved list. If one is present
#comment out if not
saved_list = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_artist.csv', engine='python', encoding='latin-1')

#grabbing only artist ids 
artist_ids = spotify.iloc[:, 5]
artist_id_array = np.array(artist_ids)

#cleaning the format of the ids to remove clutter
clean_artists =[]

for ids in artist_id_array:    
    ids = re.sub(r"\'","",ids)
    ids = re.sub(r"\[","",ids)
    ids = re.sub(r"\]","",ids)
    clean_artists.append(ids)


#isolating the main artist id to avoid errors when sending request
for i in range(len(clean_artists)):
    clean_artists[i] = clean_artists[i][:22]

import base64
import requests
import datetime
from urllib.parse import urlencode


'''please use your own spotify client ID. You can grab one here
https://developer.spotify.com/discover/'''
#insert your client id and secret in as strings
client_id = clientids.iloc[1,1]
client_secret = clientids.iloc[1,2]

client_id2 = clientids.iloc[2,1]
client_secret2 = clientids.iloc[2,2]


class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()
    
    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }
    
    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        } 
    
    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Could not authenticate client.")
            # return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token() 
        return token
    
    def get_resource_header(self):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        return headers
        
        
    def get_resource(self, lookup_id, resource_type='albums', version='v1'):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
     
    def get_album(self, _id):
        return self.get_resource(_id, resource_type='albums')
    
    def get_artist(self, _id):
        return self.get_resource(_id, resource_type='artists')
    
    def base_search(self, query_params): # type
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):  
            return {}
        return r.json()
    
    def search(self, query=None, operator=None, operator_query=None, search_type='artist' ):
        if query == None:
            raise Exception("A query is required")
        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k,v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower() == "or" or operator.lower() == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"
        query_params = urlencode({"q": query, "type": search_type.lower()})
        print(query_params)
        return self.base_search(query_params)
    

#only need to if you choose to use 2 comment one out if not
spotifyid = SpotifyAPI(client_id, client_secret)
spotifyid2 = SpotifyAPI(client_id2, client_secret2)

#uncomment if you have a saved artist list    
#artist_list = saved_list.iloc[:,1:].values.tolist()
artist_list = []
failed_index = {}
i = 0
failcount = 0

#while iteration is less than length of artist list retrieve artist information at i's index
#and add 1 to i
#if data recieved isnt empty append the artist_list and add 1 to i
#else tell me it failed and try again and add 1 to fail count for every retry
#if fail count goes over 15 try with a new client id, if that fails just append a place holder
#if iteration can be divided by 10k save if not continue with loop

while i < len(clean_artists):
    
    if spotifyid2.get_artist(clean_artists[i]) != {}:
        failcount = 0
        artist_list.append(spotifyid2.get_artist(clean_artists[i]))
        i += 1
        doneyet = i
        print('Grabbing artist: ', i)
    
    else:
        print('Failed Restarting last iteration')
        
        failcount += 1
        print('Fail Count: ', failcount)
        if failcount > 15:
            print('Trying second client')
            if spotifyid.get_artist(clean_artists[i]) != {}:
                artist_list.append(spotifyid.get_artist(clean_artists[i]))
                failcount = 0
            else:
                print('Client2 failed, appending NONE')
                artist_list.append('None')
                failed_index[i] = clean_artists[i]
                
                i += 1
                doneyet = i
                failcount = 0
    #saves every 10k interations
    if i % 10000 == 0:
        saved = pd.DataFrame(artist_list)
        saved.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_artist.csv')
        print('SAVED')
        
saved = pd.DataFrame(artist_list)
saved.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_artist.csv')
    
#extracting genres
#may have to convert your artist_list to an array
#its expecting a dict
#for each item/list(genre) in the array artist_list
#check to see if there is a place holder(None)
#if not check to see if there is actual a genre for the artist
#if there is a genre grab it, if there isnt append None instead
#can also check for popularity, just insert name of the key in the dict were genre is
#should work, has only been tested once
genre_list = []
doneyet2 = 0
for genre in artist_list:
    if genre != 'None':
        if genre['genres'][0:2] != {}:
            genre_list.append(genre['genres'][0:2])
        else:
            genre_list.append('None')
    else:
        genre_list.append('None')
    doneyet2 += 1
    print('genre extracted: ', doneyet2)
       
saved_genres = pd.DataFrame(genre_list.copy())
saved_genres.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_genres.csv')
  







