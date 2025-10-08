import json
import random
import time

import aiohttp
import asyncio
import base64
import mimetypes,imghdr
import numpy as np

def get_dict_value(obj,key,default_value=''):
    if(not key in obj):
        return default_value
    if obj[key] is None:
        return default_value
    return obj[key]

def parse_json_result(resp):
    try:
        idx1 = resp.find('{')
        idx2 = resp.rfind('}')
        if (idx1 >= 0 and idx2 >= idx1):
            s = resp[idx1:idx2 + 1]
            data = json.loads(s)
            return data
    except Exception as e:
        return {}
    return {}

def format_str(text):
    return text.replace('\n','<br/>').replace('\r','')

def standardize_text(text):
    return text.replace(',','，').replace('(','（').replace(')','）').replace(':','：').replace('\t',' ').replace('\n',' ').replace('\r','')

async def async_post(url, data, timeout=1000, retires=8):
    for i in range(retires):
        text = ''
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=timeout) as response:
                    status = response.status
                    text = await response.text()
                    return status, text
        except Exception as e:
            import traceback
            err_info = traceback.format_exc()
            print('Request ERROR', err_info)
            pass
        await asyncio.sleep(i + 1.0)
    return 500, ''

def img_to_base64(path,need_head=True):
    mime, _ = mimetypes.guess_type(path)  # 'image/png' ...
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
        if not need_head:
            return data
    return f'data:{mime};base64,{data}'  # 带格式头


def add_base64_header(b64_str: str) -> str:
    b64_str = b64_str.strip()
    header = base64.b64decode(b64_str[:44])
    fmt = imghdr.what(None, header)
    if not fmt:
        fmt = 'jpg'
    if fmt == 'jpeg':
        fmt = 'jpg'
    return f'data:image/{fmt};base64,'

def get_display_data(data):
    obj = {}
    for key in data:
        v = str(data[key])
        if len(v)<1000:
            obj[key] = data[key]
        else:
            obj[key] = str(data[key])[:20]+'...'
    return json.dumps(obj, ensure_ascii=False)

def norm_vector(vector):
    v =  vector / np.linalg.norm(vector)
    return v