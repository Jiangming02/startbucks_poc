import sys,os,time,asyncio,json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from Commons import async_post,img_to_base64

#server_url = 'http://localhost:30090/openai_vl_chat'
server_url = 'http://1.116.122.151:30090/openai_vl_chat'

async def get_image_gpt_result(contents : list,temperature=0.0):
    rdata = {'messages':[{"role":"user","content":contents}],"temperature":temperature,    "chat_template_kwargs": {"enable_thinking": False}}
    status,resp_text = await async_post(server_url,rdata,retires=5)
    json_data = json.loads(resp_text)
    if 'result' in json_data:
        return json_data['result']
    #获取失败返回None
    return None

if __name__ == '__main__':
    start_time = time.time()
    b64 = img_to_base64('../test.jpg')
    contents = [{"type":"image_url","image_url":{"url":b64}},{"type":"text","text":"这是什么？回答不超过30个字。"}]
    print(asyncio.run(get_image_gpt_result(contents)))
    print('###DONE###',round(time.time()-start_time,2),'s')