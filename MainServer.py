import base64
import pickle, time, json, asyncio, threading, toml, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from MYDB import MyDB
from Commons import img_to_base64, get_display_data, img_to_base64, norm_vector, async_post
from tools.GetTextEmbeddings import get_embeddings
from tools.GetImageEmbeddings import get_image_mbeddings
from tools.GetRerankerResult import get_reranker_results
from tools.GetGPTResult import get_gpt_result
from ContentHandle import clean_text, get_questions, summerize_text, split_text, summerize_image

configs = {}
configs['server_address'] = '0.0.0.0'
configs['server_port'] = 30097
configs['gpu_id'] = None
configs['save_data_path'] = 'save_datas.pkl'
configs['save_data_interval'] = 180

with open("config.toml", "r", encoding="utf-8") as f:
    local_configs = toml.load(f)
    for key in local_configs:
        if key in configs:
            configs[key] = local_configs[key]

print('\n#####CONFIGS#####')
for key in configs:
    print(key, configs[key])
print('#####CONFIGS DONE#####\n')

global_info = {'request_idx': 0, "text_datas_len": 0, "image_datas_len": 0, "documents_len": 0}
app = FastAPI()
save_datas = {"text_datas": [], "image_datas": [], "documents": []}
text_image_b64 = img_to_base64('./test.jpg')
text_vector_length = len(asyncio.run(get_embeddings(['你好']))[0])
image_vector_length = len(asyncio.run(get_image_mbeddings(text_image_b64)))
print("Text Vector Length", text_vector_length, "Image Vector Length", image_vector_length)

if configs['gpu_id'] is None or configs['gpu_id'] < 0:
    text_db = MyDB(text_vector_length, use_gpu=False)
    image_db = MyDB(image_vector_length, use_gpu=False)
else:
    text_db = MyDB(text_vector_length, use_gpu=True, gpu_id=int(configs['gpu_id']))
    image_db = MyDB(image_vector_length, use_gpu=True, gpu_id=int(configs['gpu_id']))

f0 = Path(configs['save_data_path'])
if f0.is_file():
    with open(configs['save_data_path'], 'rb') as f:
        save_datas = pickle.load(f)
        global_info['text_datas_len'] = len(save_datas['text_datas'])
        global_info['image_datas_len'] = len(save_datas['image_datas'])
        global_info['documents_len'] = len(save_datas['documents'])
        print('###LOADED DATA###', len(save_datas['documents']), len(save_datas['text_datas']),
              len(save_datas['image_datas']), datetime.datetime.now())


def save_data_thread():
    global global_info, save_datas
    while True:
        time.sleep(configs['save_data_interval'])
        if (not global_info['text_datas_len'] == len(save_datas['text_datas'])) or (
                not global_info['image_datas_len'] == len(save_datas['image_datas'])) or (not
        global_info['documents_len'] == len(save_datas['documents'])):
            global_info['text_datas_len'] = len(save_datas['text_datas'])
            global_info['image_datas_len'] = len(save_datas['image_datas'])
            global_info['documents_len'] = len(save_datas['documents'])
            with open(configs['save_data_path'], 'wb+') as f:
                pickle.dump(save_datas, f)
            print('###SAVE DATA###', len(save_datas['documents']), len(save_datas['text_datas']),
                  len(save_datas['image_datas']),
                  datetime.datetime.now())


save_data_thread = threading.Thread(target=save_data_thread)
save_data_thread.start()

segments_map = {}
documents_map = {}

for data in save_datas['text_datas']:
    key = data['seg_id']
    vectors = []
    if 'vector' in data:
        vectors = [data['vector']]
    if 'vectors' in data:
        vectors = data['vectors']
    segments_map[key] = data
    for vector in vectors:
        text_db.FDB.add([vector], [key])

for data in save_datas['image_datas']:
    key = data['seg_id']
    vectors = []
    if 'vector' in data:
        vectors = [data['vector']]
    if 'vectors' in data:
        vectors = data['vectors']
    segments_map[key] = data
    for vector in vectors:
        image_db.FDB.add([vector], [key])

for data in save_datas['documents']:
    key = data['doc_id']
    documents_map[key] = data


# b64 = img_to_base64('d:/imgs/cat3.jpg')
# ebd = asyncio.run(get_image_mbeddings(b64))
# ebd = ebd / np.linalg.norm(ebd)
# print(image_db.search([ebd],min_score=0))

@app.post("/get_segments")
async def get_segments(request: Request):
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST get_vectors]", request_idx, get_display_data(data))
        if 'ids' in data:
            ids = data['ids']
            results = []
            for id in ids:
                if id in segments_map:
                    obj = segments_map[id]
                    obj_clone = {}
                    for key in obj:
                        if not key == 'vector' and not key == 'vectors':
                            obj_clone[key] = obj[key]
                    results.append(obj_clone)
        else:
            doc_id = data['doc_id']
            results = []
            for obj in save_datas['text_datas']:
                if 'doc_id' in obj and obj['doc_id'] == doc_id:
                    obj_clone = {}
                    for key in obj:
                        if not key == 'vector' and not key == 'vectors':
                            obj_clone[key] = obj[key]
                    results.append(obj_clone)
        return JSONResponse(status_code=200, content={"status": 0, "result": results}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


async def do_add_data(data):
    key = data['seg_id']
    vectors = []
    if 'vector' in data:
        vectors = [data['vector']]
    if 'vectors' in data:
        vectors = data['vectors']
    if 'type' in data and data['type'] == 'image':
        for vector in vectors:
            vector = norm_vector(vector)
            image_db.FDB.add([vector], [key])
        save_datas['image_datas'].append(data)
    else:
        for vector in vectors:
            vector = norm_vector(vector)
            text_db.FDB.add([vector], [key])
        save_datas['text_datas'].append(data)
    segments_map[key] = data


@app.post("/add_data")
async def add_data(request: Request):
    global save_datas
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST add_data]", request_idx,
              get_display_data(data))
        await do_add_data(data)
        return JSONResponse(status_code=200, content={"status": 0}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


async def do_search_data_by_vector(data):
    vector = await get_embeddings([data['text']])
    if len(vector) == 1:
        vector = vector[0]
    vector = norm_vector(vector)
    top_k_embedding = 1000
    if 'top_k_embedding' in data:
        top_k_embedding = data['top_k_embedding']
    min_score = 0.2
    if 'min_score' in data:
        min_score = data['min_score']
    top_k_reranker = 20
    if 'top_k_reranker' in data:
        top_k_reranker = data['top_k_reranker']
    search_results = text_db.search([vector], top_k=top_k_embedding * 3, min_score=min_score)
    results = []
    done_keys = {}
    for arr in search_results:
        for d in arr:
            key = d['key']
            score = d['score']
            if not key in done_keys:
                results.append(d)
                done_keys[key] = d
            else:
                if score > done_keys[key]['score']:
                    done_keys[key]['score'] = score
    results.sort(key=lambda x: x['score'], reverse=True)

    results2 = []
    texts = []
    for d in results:
        obj = segments_map[d['key']]
        obj2 = {}
        for key in obj.keys():
            if not key == 'vector' and not key == 'vectors':
                obj2[key] = obj[key]
        obj2['score'] = d['score']
        results2.append(obj2)
        texts.append(obj2['text'])
        if len(results2) >= top_k_embedding:
            break
    if len(results2) == 0:
        return []

    reranker_results = await get_reranker_results(data['text'], texts, min_score=min_score, top_k=top_k_reranker)
    results3 = []
    for d in reranker_results:
        idx = d['key']
        obj = results2[idx]
        obj['score'] = d['score']
        obj_clone = {}
        for key in obj:
            if not key == 'vector' and not key == 'vectors':
                obj_clone[key] = obj[key]
        results3.append(obj_clone)
    return results3


@app.post("/search_data_by_vector")
async def search_data_by_vectors(request: Request):
    global save_datas
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST search_data_by_vector]", request_idx,
              get_display_data(data))
        results = await do_search_data_by_vector(data)
        return JSONResponse(status_code=200, content={"status": 0, "results": results}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


@app.post("/search_image_data_by_vector")
async def search_image_data_by_vector(request: Request):
    global save_datas
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST search_image_data_by_vector]", request_idx,
              get_display_data(data))
        vector1 = await get_image_mbeddings(data['base64_image'])
        vector1 = norm_vector(vector1)
        vector2 = await get_image_mbeddings(data['text'], type='text')
        vector2 = norm_vector(vector2)
        top_k_embedding = 10
        if 'top_k_embedding' in data:
            top_k_embedding = data['top_k_embedding']
        min_score = 0.1
        if 'min_score' in data:
            min_score = data['min_score']

        search_results = image_db.search([vector1, vector2], top_k=top_k_embedding * 3, min_score=min_score)
        results = []
        done_keys = {}
        for arr in search_results:
            for d in arr:
                key = d['key']
                score = d['score']
                if not key in done_keys:
                    results.append(d)
                    done_keys[key] = d
                else:
                    if score > done_keys[key]['score']:
                        done_keys[key]['score'] = score
        results.sort(key=lambda x: x['score'], reverse=True)

        results2 = []
        for d in results:
            obj = segments_map[d['key']]
            obj2 = {}
            for key in obj.keys():
                if not key == 'vector' and not key == 'vectors':
                    obj2[key] = obj[key]
            obj2['score'] = d['score']
            results2.append(obj2)
            if len(results2) >= top_k_embedding:
                break

        return JSONResponse(status_code=200, content={"status": 0, "results": results2}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


@app.post("/get_docs")
async def get_docs(request: Request):
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST get_docs]", request_idx,
              get_display_data(data))
        ids = data['ids']
        results = []
        for id in ids:
            if id in documents_map:
                obj = documents_map[id]
                results.append(obj)
        return JSONResponse(status_code=200, content={"status": 0, "result": results}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


@app.post("/add_document")
async def add_document(request: Request):
    global save_datas
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST add_document]", request_idx,
              get_display_data(data))
        doc = data['document']
        doc_id = doc['doc_id']
        doc_info = {}
        doc_info['doc_id'] = doc_id
        save_datas['documents'].append(doc)
        documents_map[doc['doc_id']] = doc
        for key in doc:
            if not key == 'contents' and not key == 'type' and not key == 'title':
                doc_info[key] = doc[key]
        if doc['type'] == 'default':
            title1 = ''
            title2 = ''
            if 'title' in doc:
                title1 = '【标题】' + doc['title'] + '\n'
                title2 = doc['title'] + '\n'
            if 'tags' in doc:
                for tag in doc['tags']:
                    title2 += tag + '\n'
            seg_index = 0
            seg_ids = []
            for cdata in doc['contents']:
                ctype = cdata['type']
                if ctype == 'image':
                    seg_index += 1
                    seg_id = doc_id + '_' + str(seg_index)
                    seg_ids.append(seg_id)
                    image_id = cdata['image_id']
                    base64_image = cdata['base64_image']
                    rdata = {"seg_id": seg_id, "doc_id": doc_id, "doc_info": doc_info, "image_id": image_id,
                             "type": "image"}
                    if 'image_url' in cdata:
                        rdata['image_url'] = cdata['image_url']
                    vectors = []
                    vector1 = await get_image_mbeddings(base64_image)
                    vectors.append(norm_vector(vector1))
                    if 'text' in cdata and cdata['text'] is not None and len(cdata['text']) > 0:
                        text = await clean_text(cdata['text'])
                        rdata['text'] = text
                        vector2 = await get_image_mbeddings(title1 + text, type='text')
                        vectors.append(norm_vector(vector2))
                    summary = await summerize_image(base64_image)
                    rdata['image_summary'] = summary
                    vector3 = await get_image_mbeddings(summary, type='text')
                    vectors.append(norm_vector(vector3))
                    rdata['vectors'] = vectors
                    await do_add_data(rdata)
                if 'text' in cdata and cdata['text'] is not None and len(cdata['text']) > 0:
                    text = await clean_text(cdata['text'])
                    if len(text) > 0:
                        text = await summerize_text(text)
                    if ctype == 'image':
                        handle_texts = [text]
                        if len(title1) > 0:
                            handle_texts.append(title1 + text)
                        vectors = await get_embeddings(handle_texts)
                        for j in range(len(vectors)):
                            vectors[j] = norm_vector(vectors[j])
                        seg_index += 1
                        seg_id = doc_id + '_' + str(seg_index)
                        seg_ids.append(seg_id)
                        rdata = {"text": text, "seg_id": seg_id, "doc_id": doc_id, "vectors": vectors,
                                 "doc_info": doc_info, "image_id": cdata['image_id'], "type": "text"}
                        if 'image_url' in cdata:
                            rdata['image_url'] = cdata['image_url']
                        await do_add_data(rdata)
                    else:
                        texts = await split_text(cdata['text'])
                        for t in texts:
                            handle_texts = [t]
                            if len(title1) > 0:
                                handle_texts.append(title1 + t)
                            if len(title2) > 0:
                                handle_texts.append(title2 + t)
                                if not title2 == title1:
                                    handle_texts.append(title2)
                            tasks = [summerize_text(t), get_questions(t)]
                            tresults = await asyncio.gather(*tasks)
                            if tresults[0] is not None and len(tresults[0]) > 0:
                                handle_texts.append(tresults[0])
                            if tresults[1] is not None and len(tresults[1]) > 0:
                                for q in tresults[1]:
                                    handle_texts.append(q)
                            vectors = await get_embeddings(handle_texts)
                            for j in range(len(vectors)):
                                vectors[j] = norm_vector(vectors[j])
                            seg_index += 1
                            seg_id = doc_id + '_' + str(seg_index)
                            seg_ids.append(seg_id)
                            rdata = {"text": title2+t, "seg_id": seg_id, "doc_id": doc_id, "vectors": vectors,
                                     "doc_info": doc_info, "type": "text"}
                            if 'image_id' in cdata:
                                rdata['image_id'] = cdata['image_id']
                            if 'image_url' in cdata:
                                rdata['image_url'] = cdata['image_url']
                            await do_add_data(rdata)
            return JSONResponse(status_code=200, content={"status": 0, "seg_ids": seg_ids},
                                media_type='application/json')
        return JSONResponse(status_code=500, content={"status": 2, "error_info": "类型暂未支持"},
                            media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        print(err_info)
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


from Chat import get_talk, get_talk_topic, get_response_talk_prompt, get_all_references


async def get_vector_search(talk, talk_round=1):
    min_score = 0.05
    top_k_embedding = 300 * round(talk_round / 2.0 + 0.1)
    if top_k_embedding>1200:
        top_k_embedding = 1200
    top_k_reranker = 15
    rdata = {"text": talk, "min_score": min_score, "top_k_embedding": top_k_embedding, "top_k_reranker": top_k_reranker}
    return await do_search_data_by_vector(rdata)


@app.post("/chat")
async def chat(request: Request):
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST chat]", request_idx,get_display_data(data))
        use_stream = False
        if 'stream' in data and data['stream']:
            use_stream = True
        chat_params = {}
        if 'chat_params' in data:
            chat_params = data['chat_params']
        personal_info = ''
        if 'personal_info' in chat_params:
            personal_info = chat_params['personal_info']
        talk = get_talk(data, personal_info=personal_info)
        tasks = []
        tasks.append(get_talk_topic(talk))
        tasks.append(get_vector_search(talk, talk_round=len(data['messages'])))
        results = await asyncio.gather(*tasks)
        if not use_stream:
            ref_docs = results[1]
            for doc in ref_docs:
                if 'text' in doc:
                    t = doc['text']
                    if t.__contains__('FY23'):
                        doc['score']+=0.05
                    if t.__contains__('FY24'):
                        doc['score']+=0.25
                    if t.__contains__('FY25'):
                        doc['score']+=0.45
                    if t.__contains__('FY26'):
                        doc['score']+=0.65
            ref_docs.sort(key=lambda x: x['score'], reverse=True)
            prompt = get_response_talk_prompt(talk, ref_docs)
            resp = await get_gpt_result(prompt, temperature=0.3)
            refs = get_all_references(resp)
            most_relate = None
            relates = []
            done_relates = set()
            for idx in refs:
                i = int(idx) - 1
                if i in done_relates:
                    continue
                done_relates.add(i)
                obj = results[1][i]
                obj['ref_link'] = '【参考资料' + idx + '】'
                if most_relate is None:
                    most_relate = obj
                relates.append(obj)
            return JSONResponse(status_code=200, content={"status": 0, "result": resp, "chat_type": results[0],
                                                          "most_related_doc": most_relate, "related_docs": relates},
                                media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=configs['server_address'], port=configs['server_port'])
