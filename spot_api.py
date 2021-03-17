import base64
import requests
import datetime
from urllib.parse import urlencode

client_id = "f8ba7e016c3548a48aad449ab1ba11aa"
client_secret = "2fabafc2dcef4001a2b4b1cc53b0cf4c"

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
            return False
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
        done_auth = self.perform_auth()
        if not done_auth:
            raise Exception("Auth Failed")
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
    def search(self, query, search_type='artist'):
        
        access_token = self.get_access_token()
        headers = {"Authorization" : f"Bearer {access_token}"}
        endpoint = "https://api.spotify.com/v1/search?"
        data = urlencode({"q" : query, "type" : search_type.lower()})#the query or item you rsearching for
        lookup_url = f"{endpoint}{data}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200,299):
            return {}
        return r.json()
'''   
spotify = SpotifyAPI(client_id, client_secret)

auth_success = spotify.search("faded alan walker", "track")

token = spotify.access_token
'''








