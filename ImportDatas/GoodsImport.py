import asyncio, time
import pandas as pd
import os
import ast
import json
import threading

import_url = 'http://localhost:30099/add_document'

activate_name = {'ly5XdtUiOz8wa7xc': 'FY23活动档期', 'aglOZaREQrwSukAz': 'FY24活动档期'}
parse_data_path = '../datas_parse_2509/'
origin_data_path = '../datas_origin/'

from tools.UpdateImage import update_image
from Commons import img_to_base64, async_post

thread_status = 0


def upload_data(data, key):
    global thread_status, import_url
    thread_status += 1
    start_time = time.time()
    print(f'###START IMPORT {key}###')
    try:
        asyncio.run(async_post(import_url, data))
    except Exception as e:
        pass
    print(f'###FINISH IMPORT {key}###', round(time.time() - start_time, 1), 's')
    thread_status -= 1


name_content_map = {}
content_name_map = {}


def parse_md_files(target_dir):
    for entry_name in os.listdir(target_dir):
        entry_path = os.path.join(target_dir, entry_name)
        if os.path.isfile(entry_path):
            # file_names.append(entry_name)  # 收集文件名（不含路径）
            name = entry_name
            if name.endswith('.md'):
                name = name[:-3]
            with open(entry_path, 'r', encoding='utf-8') as f:
                content = f.read()
                name_content_map[name] = content
                content_name_map[content] = name


parse_md_files(parse_data_path + 'md/')

content_details = {}


def parse_md_dirs(target_dir):
    for entry_name in os.listdir(target_dir):
        entry_path1 = os.path.join(target_dir, entry_name)
        if os.path.isdir(entry_path1):
            file_path = os.path.join(entry_path1, entry_name + '.md')
            name = None
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                name = content_name_map[content]
            target_dir2 = os.path.join(entry_path1, 'pages')
            for entry_name2 in os.listdir(target_dir2):
                file_path2 = os.path.join(target_dir2, entry_name2)
                if os.path.isfile(file_path2):
                    idx = entry_name2.find('.')
                    name0 = entry_name2[:idx]
                    obj = {'image_path': file_path2, 'name': entry_name2, 'target_name': name, 'code': entry_name}
                    file_path3 = os.path.join(entry_path1, 'txt_gpt')
                    file_path3 = os.path.join(file_path3, name0 + '_gpt.md')
                    if os.path.isfile(file_path3):
                        obj['md_path'] = file_path3
                    if not 'md_path' in obj:
                        print('ERROR', name, name0)
                    else:
                        if not name in content_details:
                            content_details[name] = []
                        content_details[name].append(obj)


print('###', len(name_content_map), len(content_name_map), len(content_details))

parse_md_dirs(parse_data_path + 'detail/')
check_names = set()
done_names = set()


async def do_load_task():
    global thread_status
    csv_path1 = origin_data_path + 'Goods.csv'
    csv_path2 = origin_data_path + 'Goods_origin.csv'
    wb1 = pd.read_csv(csv_path1)
    wb2 = pd.read_csv(csv_path2)
    print(len(wb1), len(wb2))
    for i in range(len(wb1)):
        gid = str(wb1['goods_id'][i]).strip()
        doc_id = 'GOODS_251007_' + gid
        good_name = str(wb1['name'][i]).strip()
        category = str(wb1['category'][i]).strip()
        if len(category) > 0 and not category == 'nan':
            good_name = good_name + '(' + category + ')'
        tags = []
        wid = str(wb1['workbook'][i])
        if wid in activate_name:
            good_name += '('+activate_name[wid]+')'
        tag_str = str(wb1['tag'][i]).strip()
        if len(tag_str) > 0 and not tag_str == 'nan':
            tags.append(tag_str)
        description = str(wb1['description'][i])
        rdoc = {"title": good_name, "doc_id": doc_id, "doc_id_origin": gid, "type": "default", "tags": tags,
                "contents": []}
        rdoc['contents'].append({"type": "text", "text": "描述或理念是:\n" + description})

        index = -1
        for j in range(len(wb2)):
            if gid == str(wb2['goods_id'][j]):
                index = j
                break
        price_l = str(wb2['lowest_price'][index])
        price_m = str(wb2['medium_price'][index])
        price_h = str(wb2['highest_price'][index])
        price_text = ''
        if not price_l == '0' and not price_l == 'nan' and not price_l == '0.0':
            price_text += '最低价格:' + price_l + '元\t'
        if not price_m == '0' and not price_m == 'nan' and not price_m == '0.0':
            price_text += '平均价格:' + price_m + '元\t'
        if not price_h == '0' and not price_h == 'nan' and not price_h == '0.0':
            price_text += '最高价格:' + price_h + '元\t'
        if len(price_text) > 0:
            rdoc['contents'].append({"type": "text", "text": "价格是:\n" + price_text})
        try:
            recipes = ast.literal_eval(wb1['recipe_card'][i])
            for recipe in recipes:
                if 'name' in recipe:
                    rname = recipe['name']
                    rname = rname.replace('?','®')
                    # if rname in check_names:
                    #     print('!!!!', rname)
                    if 'file_type' in recipe and recipe['file_type']=='jw_n_doc':
                        if not rname in content_details:
                            print('MISSING',good_name,rname,recipe['url'])
                    if rname == '祝悦杜松子风味拿铁' or rname == 'Mastrena 2代咖啡机 臻选浓缩制作指引' or rname == '太妃榛果拿铁' or rname == '星光盛典预告':
                        continue
                    check_names.add(rname)
                    done_names.add(rname)
                    if rname in content_details:
                        rlist = content_details[rname]
                        for obj in rlist:
                            fpath = obj['md_path']
                            b64 = img_to_base64(obj['image_path'])
                            image_name = str(wb1['name'][i]) + '_' + obj['code'] + '_' + obj['name']
                            image_id = doc_id + '_' + obj['target_name'] + '_' + obj['name']
                            image_url = await update_image(b64, image_name)
                            base64_url = '以下内容可以此链接查看![' + str(wb1['name'][i]) + '](' + image_url + ')\n\n'
                            with open(fpath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                cobj = {"type": "text", "text": base64_url + content, "image_id": image_id,
                                        "image_url": image_url}
                                rdoc['contents'].append(cobj)
        except Exception as e:
            pass
        try:
            img_list = ast.literal_eval(wb1['imgs'][i])
            for iobj in img_list:
                if 'url' in iobj:
                    url = iobj['url']
                    if url.__contains__('getfile') and not url.__contains__('download?'):
                        image_id = doc_id + '_' + url
                        url_content = '这是' + str(wb1['name'][i]) + '的图片![' + str(
                            wb1['name'][i]) + '](' + url + ')\n\n'
                        cobj = {"type": "text", "text": url_content, "image_id": image_id,
                                "image_url": url}
                        rdoc['contents'].append(cobj)
        except Exception as e:
            pass
        rdata = {"document": rdoc}
        info_key = str(i) + ' ' + good_name

        upload_thread = threading.Thread(target=upload_data,args=(rdata, info_key))
        upload_thread.start()
        await asyncio.sleep(0.1)

        while True:
            if thread_status >= 8:
                await asyncio.sleep(3)
            else:
                break
    wb1 = None
    wb2 = None
    wb3 = pd.read_csv(origin_data_path+'MbStore.csv')
    for i in range(len(wb3)):
        name = wb3['name'][i]
        if name=='其他陈列物-Nitro门店':
            continue
        name = name+'(摆放方式)'
        id = str(wb3['store_id'][i])
        doc_id =  'MbStore_251007_' + id
        brief = wb3['brief'][i]
        tags = []
        wid = str(wb3['workbook'][i])
        if wid in activate_name:
            name += '('+activate_name[wid]+')'
        rdoc = {"title": name, "doc_id": doc_id, "doc_id_origin": id, "type": "default", "tags": tags,
                "contents": []}
        rdoc['contents'].append({"type": "text", "text": str(brief)})
        rdata = {"document": rdoc}
        info_key = str(i) + ' ' + name

        upload_thread = threading.Thread(target=upload_data,args=(rdata, info_key))
        upload_thread.start()
        await asyncio.sleep(0.1)

        while True:
            if thread_status >= 8:
                await asyncio.sleep(3)
            else:
                break
    wb3 = None

    wb4 = pd.read_csv(origin_data_path + 'Article.csv')
    for i in range(len(wb4)):
        id = str(wb4['oid'][i])
        doc_id =  'Article_251007_' + id
        title = str(wb4['name'][i])+'('+str(wb4['summary'][i])+')'
        content = str(wb4['content'][i])
        rdoc = {"title": title, "doc_id": doc_id, "doc_id_origin": str(wb4['oid'][i]), "type": "default", "tags": [],
                "contents": []}
        rdoc['contents'].append({"type": "text", "text": content})

        rdata = {"document": rdoc}
        info_key = str(i) + ' ' + str(wb4['name'][i])

        upload_thread = threading.Thread(target=upload_data,args=(rdata, info_key))
        upload_thread.start()
        await asyncio.sleep(0.1)

        while True:
            if thread_status >= 8:
                await asyncio.sleep(3)
            else:
                break
    wb4 = None

    wb5 = pd.read_csv(origin_data_path + 'PfPoster.csv')
    for i in range(len(wb5)):
        id = str(wb5['pid'][i])
        doc_id = 'PfPoster_251007_' + id
        tags = []
        title = str(wb5['name'][i])
        if title=='nan' or title=='':
            continue
        content = str(wb5['brief'][i])
        if content=='nan' or content=='':
            continue
        wid = str(wb5['workbook'][i])
        if wid in activate_name:
            title += '(' + activate_name[wid] + ')'

        rdoc = {"title": title, "doc_id": doc_id, "doc_id_origin": id, "type": "default", "tags": tags,
                "contents": []}
        rdoc['contents'].append({"type": "text", "text": content})

        rdata = {"document": rdoc}
        info_key = str(i) + ' ' + title

        upload_thread = threading.Thread(target=upload_data,args=(rdata, info_key))
        upload_thread.start()
        await asyncio.sleep(0.1)

        while True:
            if thread_status >= 8:
                await asyncio.sleep(3)
            else:
                break
    wb5 = None

    wb6 = pd.read_csv(origin_data_path + 'BcCounter.csv')
    for i in range(len(wb6)):
        id = str(wb6['cid'][i])
        doc_id = 'BcCounter_251007_' + id
        tags = []
        wid = str(wb6['workbook'][i])
        title = str(wb6['name'][i])
        if title=='nan' or title=='':
            continue
        content = str(wb6['content'][i])
        if content=='nan' or content=='':
            continue
        if wid in activate_name:
            title += '(' + activate_name[wid] + ')'
        rdoc = {"title": title, "doc_id": doc_id, "doc_id_origin": id, "type": "default", "tags": tags,
                "contents": []}
        rdoc['contents'].append({"type": "text", "text": content})

        rdata = {"document": rdoc}
        info_key = str(i) + ' ' + title

        upload_thread = threading.Thread(target=upload_data,args=(rdata, info_key))
        upload_thread.start()
        await asyncio.sleep(0.1)

        while True:
            if thread_status >= 8:
                await asyncio.sleep(3)
            else:
                break

    for name in content_details:
        if not name in done_names:
            olist = content_details[name]
            doc_id = '251007_'+name
            rdoc = {"title": name, "doc_id": doc_id, "doc_id_origin": name, "type": "default", "tags": [],
                    "contents": []}
            for obj in olist:
                fpath = obj['md_path']
                b64 = img_to_base64(obj['image_path'])
                image_name = str(name)+'_'+obj['name']
                image_id = doc_id+'_'+obj['target_name'] + '_' + obj['name']
                image_url = await update_image(b64, image_name)
                base64_url = '以下内容可以此链接查看![' + image_name + '](' + image_url + ')\n\n'
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    cobj = {"type": "text", "text": base64_url + content, "image_id": image_id,
                            "image_url": image_url}
                    name0 = obj['name']
                    idx = name0.find('.')
                    if idx>0:
                        cobj['page_index'] = name0[:idx]
                    rdoc['contents'].append(cobj)

            rdata = {"document": rdoc}
            info_key = name

            upload_thread = threading.Thread(target=upload_data,args=(rdata, info_key))
            upload_thread.start()
            await asyncio.sleep(0.1)

            while True:
                if thread_status >= 8:
                    await asyncio.sleep(3)
                else:
                    break


    while True:
        if thread_status > 0:
            print('剩余任务',thread_status)
            await asyncio.sleep(10)
        else:
            break

if __name__ == '__main__':
    start_time = time.time()
    asyncio.run(do_load_task())
    print('###ALL DONE###',round(time.time()-start_time,1),'s')
