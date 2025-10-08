import asyncio

from tools.GetGPTResult import get_gpt_result
from tools.GetImageGPTResult import get_image_gpt_result
from Commons import parse_json_result,img_to_base64

async def clean_text(text):
    if text.__contains__('</p>') or text.__contains__('</span>'):
        pass
    else:
        return text
    prompt = '以下是一段网页文本:\n'
    prompt += text
    prompt += '\n###网页结束###'
    prompt += '\n\n请将以上网页文本转换成Markdown格式的文本并输出，具体要求如下：'
    prompt += '\n1、转换后只需要保留表格样式和段落样式，其它所有样式比如字体、背景等可以删除。'
    prompt += '\n2、不同段落间要有明显的分隔。'
    prompt += '\n3、直接输出转换后的文本，不需要输出其他额外说明。'
    try:
        gpt_result = await get_gpt_result(prompt)
        if len(gpt_result)>0:
            return gpt_result
    except Exception as e:
        pass
    return text

max_doc_len = 1000
cross_len = 50
async def split_text(text : str):
    text0 = text[:].strip()
    if len(text0)<=max_doc_len:
        return [text0]
    start_index = 0
    result = []
    while True:
        end_index = start_index + max_doc_len
        if end_index >= len(text0):
            result.append(text0[start_index:])
            break
        result.append(text0[start_index:end_index])
        start_index = start_index + max_doc_len - cross_len
    return result

async def summerize_text(text):
    prompt = '以下是一段文本:\n'+text+'\n###文本结束###'
    if len(text)<=100:
        prompt += '\n\n概述下以上的文本讲了什么，不超过100字。除了概述，不需要输出其它内容。'
    else:
        prompt += '\n\n概述下以上的文本讲了什么，通常50-100字。除了概述，不需要输出其它内容。'
    try:
        gpt_result = await get_gpt_result(prompt)
        if len(gpt_result) > 0:
            return gpt_result
    except Exception as e:
        pass
    return None

async def summerize_image(base64_image):
    contents = [{"type":"image_url","image_url":{"url":base64_image}},{"type":"text","text":"概述下这张图片，要保留独特细节，不超过200字。"}]
    try:
        gpt_result = await get_image_gpt_result(contents)
        if len(gpt_result) > 0:
            return gpt_result
    except Exception as e:
        pass
    return None

async def get_questions(text):
    prompt = '以下是一段文本:\n' + text + '\n###文本结束###'
    prompt += '\n根据以上文本，请你出最多10个问题，具体要求如下：'
    prompt += '\n1、以json的形式输出结果，输出的结果必须能被python中的json.loads函数读取。只需要输出json，不需要输出其它文字或者说明。'
    prompt += '\n2、输出的json有一个key"问题列表"，类型为数组，里面的值都是字符串。'
    prompt += '\n3、问题的主语要明确。问题要优先诊断里面的段落而非个别词。'
    prompt += '\n4、严格按照以下json形式输出结果：{"问题列表":[]}'
    try:
        gpt_result = await get_gpt_result(prompt)
        if len(gpt_result) > 0:
            jdata = parse_json_result(gpt_result)
            if '问题列表' in jdata:
                return jdata['问题列表']
    except Exception as e:
        pass
    return []

if __name__ == '__main__':
    # text = '123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890'
    # max_doc_len = 8
    # cross_len = 2
    # print(asyncio.run(split_text(text)))
    text = '''## 附录：门店工具– 饮料原料QA注意事项
### SKU 11143070 太妃糖混合榛子粒
- **图片**：  
- **饮品**：
  - 太妃榛果拿铁
  - 太妃榛果红茶拿铁
  - 星巴克太妃榛果冰震浓缩
  - 太妃榛果雪融拿铁
- **应产率**：
  - 中杯：42
  - 大杯：42
  - 超大杯：42
- **最小订货单位**：1袋（50克/袋）
- **开封后保质期**：
  - 原包装：到店后未开封需进冰箱冷藏
  - 转容器：5天，冷藏
- **订货信息**：主配，预计分6-8批到店，具体信息可留意后续每周订货沟通'''
    print(asyncio.run(summerize_text(text)))
    print(asyncio.run(get_questions(text)))
    print(asyncio.run(clean_text('<p>你好</p>')))
    b64 = img_to_base64('d:/imgs/cat1.jpg')
    print(asyncio.run(summerize_image(b64)))