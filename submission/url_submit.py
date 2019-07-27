# coding: utf-8
import requests
files={'files': open('/home/fly/PycharmProjects/other_people_baseline/DS420-Cobras-master/Submissions/RandomForest_11_5_2018_3_48_27.csv','rb')}
data = {    "user_id": "m_18437123536",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
            "team_token": "6b2cd3e83feda127ef1ebb47325973ec0a2d0954a38c3a8b2d09db584979f0dd", #your team_token.
            "description": 'ok',  #no more than 40 chars.
            "filename": "my_submissioin", #your filename
       }
url = 'https://biendata.com/competition/kdd_2018_submit/'
response = requests.post(url, files=files, data=data)
print(response.text)