import asyncio

from tools.GetGPTResult import get_gpt_result
from tools.GetImageGPTResult import get_image_gpt_result
from Commons import parse_json_result
import re

def get_talk_topic_prompt(talk):
    prompt = '以下是我和用户的对话:\n'+talk+'\n###对话结束###\n\n'
    prompt += '请你判断用户最后的对话想要解决什么问题，具体要求如下：'
    prompt += '\n1、以json的形式输出结果，输出的结果必须能被python中的json.loads函数读取。只需要输出json，不需要输出其它文字或者说明。'
    prompt += '\n2、输出的json有一个key"对话主题“，类型为字符串,值为"活动信息查询"、"产品操作查询"和"其它"之一。'
    prompt += '\n3、和活动有关的，包括活动产品都属于"活动查询"。'
    prompt += '\n4、和活动无关的产品、操作的查询，属于"产品操作查询"。'
    prompt += '\n5、严格按照以下json形式输出结果：{"对话主题":""}'
    return prompt

async def get_talk_topic(talk):
    talk_topic_prompt = get_talk_topic_prompt(talk)
    resp = await get_gpt_result(talk_topic_prompt)
    jdata = parse_json_result(resp)
    if "对话主题" in jdata:
        topic = str(jdata['对话主题']).strip()
        if topic.__contains__('活动查询'):
            return '活动查询'
        if topic.__contains__('产品操作查询'):
            return '数据库查询'
    return "混合查询"

def get_talk(openai_data,personal_info=''):
    text = ''
    if len(personal_info)>0:
        text = '用户的背景信息如下：'+personal_info.replace('\r','').replace('\n','<br>')+'\n\n'
    for data in openai_data['messages']:
        if data['role']=='user':
            text += '用户说：'+data['content'].replace('\r','').replace('\n','<br>')+'\n'
        else:
            text += '我回答：' + data['content'].replace('\r', '').replace('\n', '<br>') + '\n'
    return text

def get_response_talk_prompt(talk,docs):
    prompt = ''
    if len(docs)>0:
        prompt = '以下是数据库中查询到的参考资料：\n'
    for i in range(len(docs)):
        prompt += '###参考资料'+str(i+1)+'###\n'
        prompt += docs[i]['text']
        prompt += '###参考资料' + str(i+1) + '结束###\n'
    if len(docs) > 0:
        prompt+= '###所有参考资料结束###\n\n'
    prompt += '以下是我和用户的对话:\n'+talk+'\n###对话结束###\n\n'
    if len(docs) > 0:
        prompt+= '需要你根据参考资料，回答用户问题。具体要求如下:\n'
        prompt += '\n1、优先根据参考资料中的信息回答。'
        prompt += '\n2、如果参考资料中没有，回答开始的时候先说明"数据库中未查询到相关信息，建议咨询相关人员。"，然后帮助用户解决问题。'
        prompt += '\n3、在参考资料无法命中的情况下，常识性问题、开放性问题、与活动和产品无关的问题你可以尝试性回答。'
        prompt += '\n4、回复中要有优先输出最可能的答案。'
        prompt += '\n5、如果回答中有引用参考资料中的图片链接，需要以MARKDOWN的形式输出图片。'
        prompt += '\n6、如果有多份相关文档，参考和引用的优先级为"FY26活动档期"、"FY25活动档期"、"FY24活动档期"、"FY23活动档期"、"FY22活动档期"。'
        prompt += '\n7、如果引用了参考资料，需要在输出时进行引用，引用标识为【参考资料XXX】,例如引用了"参考资料1"则为"【参考资料1】"。所有引用的资料（包含图片、表格）都需要标注引用标识，但同一份参考资料的引用标识最多出现一次。'
        prompt += '\n8、直接回复用户的对话，不需要输出其它文字。'
    else:
        prompt += '需要你直接回答用户问题。具体要求如下:'
        prompt += '\n1、优先根据参考资料中的信息回答。'
        prompt += '\n2、由于数据库中未查询到任何参考资料，回答开始的时候先说明"数据库中未查询到相关信息，建议咨询相关人员。"，然后帮助用户解决问题。'
        prompt += '\n3、常识性问题、开放性问题、与活动和产品无关的问题你可以尝试性回答。'
        prompt += '\n4、直接回复用户的对话，不需要输出其它文字。'
    return prompt


def get_all_references(text):
    pattern = r'【参考资料(\d{1,3})】'
    matches = re.findall(pattern, text)
    return matches