import sys,os,time,asyncio,json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from Commons import async_post

#server_url = 'http://localhost:30090/openai_chat'
server_url = 'http://1.116.122.151:30090/openai_chat'

async def get_gpt_result(prompt : str,temperature=0.0):
    rdata = {'messages':[{"role":"user","content":prompt}],"temperature":temperature,    "chat_template_kwargs": {"enable_thinking": False}}
    status,resp_text = await async_post(server_url,rdata,retires=5)
    json_data = json.loads(resp_text)
    if 'result' in json_data:
        return json_data['result']
    #获取失败返回None
    return None

if __name__ == '__main__':
    start_time = time.time()
    print(asyncio.run(get_gpt_result('你好')))
    print('###DONE###',round(time.time()-start_time,2),'s')