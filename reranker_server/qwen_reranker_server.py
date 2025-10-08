import logging
from typing import Dict, Optional, List

import json
import logging

import torch

from modelscope import AutoTokenizer, is_torch_npu_available
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from vllm.inputs.data import TokensPrompt
import os

from fastapi import FastAPI, Request
import datetime
from fastapi.responses import JSONResponse, Response
import toml

server_host = '0.0.0.0'
server_port = 30096
app = FastAPI()


configs = {}
configs['server_address'] = '0.0.0.0'
configs['server_port'] = 30097
configs['model_path'] = None
configs['gpu_id'] = 0
configs['gpu_mem'] = 0.9
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

def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
suffix_tokens = None
max_len = configs['max_model_len']

def process_inputs(tokenizer,pairs, instruction, max_length, suffix_tokens):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages =  tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        token_count = len(outputs[i].outputs[0].token_ids)
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores


class reranker:
    def __init__(self):
        global suffix_tokens,suffix,max_len,configs
        tokenizer = AutoTokenizer.from_pretrained(configs['model_path'])
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        os.environ["CUDA_VISIBLE_DEVICES"] = str(configs['gpu_id'])
        model = LLM(model=configs['model_path'], tensor_parallel_size=1, max_model_len=max_len,
                    enable_prefix_caching=True, gpu_memory_utilization=configs['gpu_mem'])
        self.tokenizer = tokenizer
        self.model = model
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        self.true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1,logprobs=20, allowed_token_ids=[self.true_token, self.false_token ])

    def get_rerank_result(self, query, texts, min_score=-999, top_k=5,keys=None):
        global suffix_tokens, suffix, max_len
        task_text = 'Given a web search query, retrieve relevant passages that answer the query'
        pairs = []
        for text in texts:
            pairs.append(format_instruction(task_text, query, text))
        inputs = process_inputs(self.tokenizer,pairs, task_text, max_len-len(suffix_tokens), suffix_tokens)
        scores = compute_logits(self.model, inputs, self.sampling_params, self.true_token, self.false_token)
        destroy_model_parallel()
        results0 = []
        for i in range(len(texts)):
            obj = {"score": float(scores[i]), "text": texts[i], "key": None}
            if keys is not None:
                obj['key'] = keys[i]
            else:
                obj['key'] = i
            if obj['score'] >= min_score:
                results0.append(obj)
        results0.sort(key=lambda x: x['score'], reverse=True)
        results1 = []
        for i in range(len(results0)):
            results1.append(results0[i])
            if len(results1) == top_k:
                break

        return results1
global_info = {'request_idx': 0}
rr = reranker()
@app.post("/reranker")
async def get_rerank_result(request: Request):
    global rr
    request_idx = global_info['request_idx']
    global_info['request_idx'] += 1
    try:
        data = await request.json()
        print(datetime.datetime.now(), "INFO[RESPONSE REQUEST get_vectors]", request_idx,
              json.dumps(data, ensure_ascii=False))
        texts = data['texts']
        query = data['query']
        min_score = -99999
        top_k = 5
        if 'min_score' in data:
            min_score = data['min_score']
        if 'top_k' in data:
            top_k = data['top_k']
        result = rr.get_rerank_result(query, texts, top_k=top_k, min_score=min_score)
        return JSONResponse(status_code=200, content={"status":0,"result":result}, media_type='application/json')
    except Exception as e:
        import traceback
        err_info = traceback.format_exc()
        return JSONResponse(status_code=500, content={'status': 1, 'error_info': str(err_info)},
                            media_type='application/json')
if __name__ == '__main__':
    query = '胸痛'
    texts = ['轻微胸痛', '小狗', '咳嗽', '阿莫西林', '疼痛']
    print('TEST RESULT',rr.get_rerank_result(query, texts, top_k=3, min_score=0))
    print('---DONE---')
    import uvicorn
    uvicorn.run(app, host=configs['server_address'], port=configs['server_port'])

