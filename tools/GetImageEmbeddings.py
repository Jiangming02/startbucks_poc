import sys,os,time,asyncio,json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from Commons import async_post,img_to_base64

#server_url = 'http://localhost:30090/get_image_vector'
server_url = 'http://1.116.122.151:30090/get_image_vector'

async def get_image_mbeddings(content : str,type : str='image'):
    if type=='text':
        rdata = {"text":content}
    else:
        rdata = {'base64_image':content}
    status,resp_text = await async_post(server_url,rdata,retires=5)
    json_data = json.loads(resp_text)
    if 'result' in json_data:
        return json_data['result']
    #获取失败返回None
    return None

if __name__ == '__main__':
    start_time = time.time()
    b64 = img_to_base64('../test.jpg')
    print(asyncio.run(get_image_mbeddings(b64)))
    print('###DONE###',round(time.time()-start_time,2),'s')