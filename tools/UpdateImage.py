import sys,os,time,asyncio,json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from Commons import async_post,img_to_base64,parse_json_result

url = ''

async def update_image(base64_image,file_name):
    if len(url) == 0:
        return 'http://test_url.com/'+file_name
    rdata = {"attachments":{"name":file_name,"datas":base64_image,"is_public":True}}
    status_code,resp = await async_post(url,rdata,retires=5)
    jdata = parse_json_result(resp)
    return jdata['data']['attachments'][0]['preview_url']

if __name__ == '__main__':
    start_time = time.time()
    b64 = img_to_base64('../test.jpg')
    print(len(b64))
    print(asyncio.run(update_image(b64,"test.jpg")))
    print('###DONE###',round(time.time()-start_time,2),'s')