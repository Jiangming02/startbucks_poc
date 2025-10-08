import sys,os,time,asyncio,json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from Commons import async_post

#server_url = 'http://localhost:30090/get_vectors'
server_url = 'http://1.116.122.151:30090/get_vectors'

async def get_embeddings(texts : list,is_query : bool = False):
    rdata = {'texts':texts,'is_query':is_query}
    status,resp_text = await async_post(server_url,rdata,retires=5)
    json_data = json.loads(resp_text)
    if 'result' in json_data:
        return json_data['result']
    #获取失败返回None
    return None

if __name__ == '__main__':
    start_time = time.time()
    print(asyncio.run(get_embeddings(['你好','hello'])))
    print('###DONE###',round(time.time()-start_time,2),'s')