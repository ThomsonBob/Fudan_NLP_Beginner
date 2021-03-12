import requests
from datetime import datetime
import random

def get_rand_temp():
    return round(random.uniform(36.8, 37.2), 1)

class NUSHTD:
    def __init__(self, username, password):
        self.session = requests.Session()

        # specifying request url & payload
        response_type = "code"
        client_id = "97F0D1CACA7D41DE87538F9362924CCB-184318"
        resource = "sg_edu_nus_oauth"
        redirect_uri = "https://myaces.nus.edu.sg:443/htd/htd"
        auth_data = {
            "UserName": username,
            "password": password,
            "authMethod": "FormsAuthentication"
        }
        auth_url = f"https://vafs.nus.edu.sg/adfs/oauth2/authorize?response_type={response_type}&client_id={client_id}&resource={resource}&redirect_uri={redirect_uri}"
        response = self.session.post(auth_url, data=auth_data)
        # check for logged in status
        if response.url.startswith(redirect_uri):
            self.isLoggedIn = True
            return
        self.isLoggedIn = False

    def declare(self, temperate, isAM=True, haveSymptoms=False, haveFamilySymptoms=False):
        # check if user is authenticated
        if not self.isLoggedIn:
            return False, "Not logged in."

        declare_url = "https://myaces.nus.edu.sg/htd/htd"
        declare_data = {
            "actionName": "dlytemperature",
            "webdriverFlag": "",
            "tempDeclOn": datetime.today().strftime("%d/%m/%Y"),
            "declFrequency": "A" if isAM else "P",
            "symptomsFlag": "Y" if haveSymptoms else "N",
            "familySymptomsFlag": "Y" if haveFamilySymptoms else "N",
            "temperature": temperate
        }
        response = self.session.post(declare_url, data=declare_data)
        if response.status_code == 200:
            return True, f'Successfully declared temperate on {declare_data["tempDeclOn"]}({declare_data["declFrequency"]}M) with {declare_data["temperature"]} degree celsius'
        return False, f'Unable to declare temperate.'
if __name__ == "__main__":
    nus_htd = NUSHTD("nusstu\E0675913", "175078Tb!@#")
    tmp=get_rand_temp()
    result, msg = nus_htd.declare(tmp)
    print(msg)
