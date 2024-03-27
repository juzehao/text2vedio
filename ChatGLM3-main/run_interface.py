from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

app = Flask(__name__)

# 加载tokenizer和模型并将模型设置为评估模式
tokenizer = AutoTokenizer.from_pretrained("/home/jzh/ai/ChatGLM3-main/THUDM/Users/zehaoju/chatglm3-6b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/jzh/ai/ChatGLM3-main/THUDM/Users/zehaoju/chatglm3-6b", trust_remote_code=True).to('cuda')
model.eval()

def parse_text(text):
    # 使用正则表达式来找到 "第n部分文本：" 和 "图片描述：" 之间的所有文本（含大括号内的文本）
    text_parts = re.findall(r'部分文本：(.*?)，图片描述：', text, re.DOTALL)

    # 清理每个文本段落，移除开头和结尾的空白字符，以及去除大括号
    text_parts = [re.sub(r'[{}]', '', part).strip().replace('\n', ' ') for part in text_parts]

    # 使用正则表达式找到所有"图片描述："后面直到句号"."的文本
    image_descriptions = re.findall(r'图片描述：(.*?。)', text, re.DOTALL)

    # 清理每个图片描述，移除开头和结尾的空白字符，以及去除换行符
    image_descriptions = [desc.strip().replace('\n', ' ') for desc in image_descriptions]

    return text_parts, image_descriptions
def solve(parts, txt):
    # 禁用梯度计算以节省显存
    with torch.no_grad():
        response, history = model.chat(tokenizer, "将这段小说转变为第一人称小说，小说如下，保留人物之间的对话：" + txt, history=[], temperature=0.1, max_length=10000)
        response2, history = model.chat(tokenizer, "将这段小说划分为" + parts + "个部分，每个部分对应一个背景图片，你需要对该背景图片进行描述，描述语句包括：图片中人物的形象，动作，神态，所处的场景，描述语句不要出现对话。输出的格式为：{第n部分文本：文本 图片描述：第n部分的背景描述语句}，以下为小说：" + response, history=[], temperature=0.1, max_length=10000)
        text_parts, image_descriptions=parse_text(response2)
        result=[]
        for i in range(len(image_descriptions)):
            response, history = model.chat(tokenizer, "将以下这段话翻译为英文,只给我英文就行：" + image_descriptions[i], history=[], temperature=0.1, max_length=10000)
            result.append(response)
        return text_parts,result

@app.route('/process_strings', methods=['POST'])
def process_strings():
    data = request.json

    if not ('parts' in data and 'text' in data):
        return jsonify({'error': 'Bad request', 'message': 'The request should contain "parts" and "text".'}), 400

    parts = data.get('parts')
    text = data.get('text')
    parts, picture_descriptions = solve(parts, text)

    return jsonify({'parts': parts, 'picture_descriptions': picture_descriptions})

if __name__ == '__main__':
    # 启动单线程的 Flask 应用
    app.run(debug=False, host='0.0.0.0', port=8080, threaded=False)