from sentence_transformers import SentenceTransformer
import os
import torch
import numpy as np
import toml

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn
import json
import datetime
import time

configs = {}
configs['server_address'] = '0.0.0.0'
configs['server_port'] = 30097
configs['model_path'] = None
configs['gpu_id'] = 0
configs['max_model_len'] = 8000

with open("config.toml", "r", encoding="utf-8") as f:
    local_configs = toml.load(f)
    for key in local_configs:
        if key in configs:
            configs[key] = local_configs[key]

print('\n#####CONFIGS#####')
for key in configs:
    print(key, configs[key])
print('#####CONFIGS DONE#####\n')

class embedding:
    def __init__(self, embedding_model,gpu_id=None):
        if(not gpu_id is None and not len(gpu_id)==0):
            device = torch.device("cuda:"+str(gpu_id))
        else:
            device = torch.device("cpu")
        model = SentenceTransformer(embedding_model, device=device)
        self.model = model

    def get_vectors(self, texts,is_query=False):
        if is_query:
            emds = self.model.encode(texts,prompt_name="query")
        else:
            emds = self.model.encode(texts)
        return emds

    def get_vectors_norm(self, texts):
        emds = self.get_vectors(texts)
        for i in range(0, len(emds)):
            emd = emds[i]
            emd = emd / np.linalg.norm(emd)
            emds[i] = emd
        return emds

    def get_vectors_similarity(self,vectors1,vectors2):
        similarity = self.model.similarity(vectors1, vectors2)
        return similarity

    def get_similarity(self,text1,text2):
        vectors1 = self.get_vectors([text1])
        vectors2 = self.get_vectors([text2])
        similarity = self.get_vectors_similarity(vectors1,vectors2)
        return float(similarity[0][0])

global_info = {'request_idx': 0}
ebd = embedding(configs['model_path'], gpu_id=str(configs['gpu_id']))
app = FastAPI()

@app.post("/embedding")
async def get_embedding_result(request: Request):
    global rr
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST get_vectors]", request_idx,
              json.dumps(data, ensure_ascii=False))
        is_query = False
        if 'is_query' in data:
            is_query = data['is_query']
        vectors = ebd.get_vectors(data['texts'], is_query=is_query)
        vectors2 = vectors.astype(float).tolist()
        return JSONResponse(status_code=200, content={"status":0,"result":vectors2}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')

if(__name__=='__main__'):
    start_time = time.time()
    print('###START TEST###')
    print(ebd.get_vectors(['你好']))
    print(ebd.get_vectors_norm(['你好']))
    print(ebd.get_similarity('你好','hello'))
    print('###TEST DONE###',round(time.time()-start_time,2))
    uvicorn.run(app, host=configs['server_address'], port=configs['server_port'])
    print('---DONE---')
