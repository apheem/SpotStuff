import numpy as np
import pandas as pd
import re
#my client id on my computer
clientids = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\clientids.csv')

spotify = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\tracks_features.csv', engine='python', encoding='latin-1')
#building a smaller dataframe that is smaller and easier to work with for ranking
spotify_campare = spotify[['name','artists']]
saved_list = pd.read_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_artist.csv', engine='python', encoding='latin-1')

artist_ids = spotify.iloc[:, 5]
artist_id_array = np.array(artist_ids)


clean_artists =[]

for ids in artist_id_array:    
    ids = re.sub(r"\'","",ids)
    ids = re.sub(r"\[","",ids)
    ids = re.sub(r"\]","",ids)
    clean_artists.append(ids)



for i in range(len(clean_artists)):
    clean_artists[i] = clean_artists[i][:22]

import base64
import requests
import datetime
from urllib.parse import urlencode


'''please use your own spotify client ID. You can grab one here
https://developer.spotify.com/discover/'''
client_id = clientids.iloc[1,1]
client_secret = clientids.iloc[1,2]

client_id2 = clientids.iloc[2,1]
client_secret2 = clientids.iloc[2,2]

# class SpotifyAPI(object):
#     access_token = None
#     access_token_expires = datetime.datetime.now()
#     access_token_did_expire = True
#     client_id = None
#     client_secret = None
#     token_url = "https://accounts.spotify.com/api/token"
    
#     def __init__(self, client_id, client_secret, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.client_id = client_id
#         self.client_secret = client_secret

#     def get_client_credentials(self):
#         """
#         Returns a base64 encoded string
#         """
#         client_id = self.client_id
#         client_secret = self.client_secret
#         if client_secret == None or client_id == None:
#             raise Exception("You must set client_id and client_secret")
#         client_creds = f"{client_id}:{client_secret}"
#         client_creds_b64 = base64.b64encode(client_creds.encode())
#         return client_creds_b64.decode()
    
#     def get_token_headers(self):
#         client_creds_b64 = self.get_client_credentials()
#         return {
#             "Authorization": f"Basic {client_creds_b64}"
#         }
    
#     def get_token_data(self):
#         return {
#             "grant_type": "client_credentials"
#         } 
    
#     def perform_auth(self):
#         token_url = self.token_url
#         token_data = self.get_token_data()
#         token_headers = self.get_token_headers()
#         r = requests.post(token_url, data=token_data, headers=token_headers)
#         if r.status_code not in range(200, 299):
#             return False
#         data = r.json()
#         now = datetime.datetime.now()
#         access_token = data['access_token']
#         expires_in = data['expires_in'] # seconds
#         expires = now + datetime.timedelta(seconds=expires_in)
#         self.access_token = access_token
#         self.access_token_expires = expires
#         self.access_token_did_expire = expires < now
#         return True
    
#     def get_access_token(self):
#         done_auth = self.perform_auth()
#         if not done_auth:
#             raise Exception("Auth Failed")
#         token = self.access_token
#         expires = self.access_token_expires
#         now = datetime.datetime.now()
#         if expires < now:
#             self.perform_auth()
#             return self.get_access_token()
#         elif token == None:
#             self.perform_auth()
#             return self.get_access_token()
#         return token
#     def search(self, query, search_type='track'):
        
#         access_token = self.get_access_token()
#         headers = {"Authorization" : f"Bearer {access_token}"}
#         endpoint = "https://api.spotify.com/v1/search?"
#         data = urlencode({"q" : query, "type" : search_type.lower()})#the query or item you rsearching for
#         lookup_url = f"{endpoint}{data}"
#         r = requests.get(lookup_url, headers=headers)
#         if r.status_code not in range(200,299):
#             return {'error'}
#         return r.json()
    
#     def artists(self, artist_id, search_type='artists'):
        
#         access_token = self.get_access_token()
#         headers = {"Authorization" : f"Bearer {access_token}"}
#         endpoint = "https://api.spotify.com/v1/tracks?"
#         data = urlencode({"q" : artist_id, "type" : search_type.lower()})#the query or item you rsearching for
#         lookup_url = f"{endpoint}{data}"
#         r = requests.get(lookup_url, headers=headers)
#         if r.status_code not in range(200,299):
#             return {'error'}
#         return r.json()
  



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
    


spotifyid = SpotifyAPI(client_id, client_secret)
spotifyid2 = SpotifyAPI(client_id2, client_secret2)

# artist_list = []
# doneyet = 0
# for artist in clean_artists:  
    
#     doneyet += 1
#     print('Grabbing artist: ', doneyet)
    
artist_list = saved_list.iloc[:,1].values.tolist()
failed_index = {}
i = len(artist_list)
failcount = 0

print(artist_list[-7:])

while i < len(clean_artists):
    
    if spotifyid.get_artist(clean_artists[i]) != {}:
        failcount = 0
        artist_list.append(spotifyid.get_artist(clean_artists[i]))
        i += 1
        doneyet = i
        print('Grabbing artist: ', i)
    
    else:
        print('Failed Restarting last iteration')
        
        failcount += 1
        print('Fail Count: ', failcount)
        if failcount > 15:
            print('Trying second client')
            if spotifyid2.get_artist(clean_artists[i]) != {}:
                artist_list.append(spotifyid2.get_artist(clean_artists[i]))
                failcount = 0
            else:
                print('Client2 failed, appending NONE')
                artist_list.append('None')
                failed_index[i] = clean_artists[i]
                
                i += 1
                doneyet = i
                failcount = 0
    if i % 50000 == 0:
        saved = pd.DataFrame(artist_list)
        saved.to_csv('C:\\Users\\cody1\\Downloads\\py_tut\\rankers\\saved_artist.csv')
        print('SAVED')
        

# for i in range(doneyet, len(clean_artists)):
#     if spotifyid.get_artist(clean_artists[i]) != {}:
#         artist_list.append(spotifyid.get_artist(clean_artists[i]))
#         doneyet = i
#         print('Grabbing artist: ', doneyet) 
#     else:
#         i -= 1
#     print(i)
    
    
genre_list = []
doneyet2 = 0
for genre in artist_list:
    if genre != {}:
        genre_list.append(genre['genres'][0:2])
    else:
        genre_list.append(0)
    doneyet2 += 1
    print('genre extracted: ', doneyet2)
       
save_artist = artist_list.copy()
    


spotifyid.search(query='take me to church', search_type="track")
token = spotify.access_token





