import requests
import random
import json
from hashlib import md5


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


# translate with google api
def translate_google(content):
    url = "https://translation.googleapis.com/language/translate/v2"
    data = {
        'key': "YOUR_API_KEY",
        'source': "zh",
        'target': "en",
        'q': content,
        'format': 'text'
    }
    headers = {'X-HTTP-Method-Override': 'GET'}
    response = requests.post(url, data=data, headers=headers)
    res = response.json()
    text = res["data"]["translations"][0]["translatedText"]

    return text


def translate_baidu(query, from_lang, to_lang):
    appid = 'YOUR APP ID'
    appkey = 'YOU APP KEY'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    return result["trans_result"][0]['dst']
