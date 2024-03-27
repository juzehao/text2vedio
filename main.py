import io
import cv2
import base64
import requests
from PIL import Image
from flask import Flask, render_template, request, send_file, jsonify
from moviepy.editor import ImageClip, AudioFileClip,VideoFileClip, concatenate_videoclips

"""
    To use this example make sure you've done the following steps before executing:
    1. Ensure automatic1111 is running in api mode with the controlnet extension. 
       Use the following command in your terminal to activate:
            ./webui.sh --no-half --api
    2. Validate python environment meet package dependencies.
       If running in a local repo you'll likely need to pip install cv2, requests and PIL 
"""


class ControlnetRequest:
    def __init__(self):
        self.url = "http://0.0.0.0:7892/sdapi/v1/txt2img"
        self.prompt = ''
        self.img_path = ['','']
        self.body = None

    def build_body(self,num):
        if num==0:
                self.body = {
                "prompt": self.prompt,
                "negative_prompt": "",
                "batch_size": 1,
                "steps": 30,
                "cfg_scale": 7,
                "alwayson_scripts": {
                    }
                }
        if num==1:
            self.body = {
            "prompt": self.prompt,
            "negative_prompt": "",
            "batch_size": 1,
            "steps": 30,
            "cfg_scale": 7,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module":"canny", 
                            "model":"control_v11p_sd15_canny_fp16 [b18e0966]",
                            "weight": 0.8,
                            "image": self.read_image(self.img_path[1]),
                            "resize_mode": 1,
                            "lowvram": False,
                            "processor_res": 64,
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 0,
                            "pixel_perfect": True
                        },
                        {

                            "enabled": True,
                            "module": "ip-adapter_clip_sd15",
                            "model": "ip-adapter_sd15_plus [32cd8f7f]",
                            "weight": 1.0,
                            "image": self.read_image(self.img_path[0]),
                            "resize_mode": 1,
                            "lowvram": False,
                            "processor_res": 64,
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 0,
                            "pixel_perfect": True
                        }
                    ]
                }
            }
        }

    def send_request(self):
        response = requests.post(url=self.url, json=self.body)
        return response.json()

    def read_image(self,path):
        img = cv2.imread(path)
        retval, bytes = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes).decode('utf-8')
        return encoded_image
def getText():
    data = {
    'parts': '5',
    'text': '什，什么？竟然有三年级能够越级挑战六年级的学生，要知道六年级可能最接近的初中的存在，而且小学每个年级之间差的不只一星半点，竟然有三年级的学生敢挑战就六年级的学生，他是不想活了吗？\"在李村唯一的小学中，此时一大一小两个学生正在对峙，他们各自守着自己的课桌，强大的气场从他们身上散发出来。只见高的那个学生负手而立，面露不屑，用虽然不大，但是能够让所有人都能够听清的普通话，说道，\"就是你，想要跟我挑战吗？我可是六年级，你要想好了。\"围观的众人纷纷惊叹于他的普通话，要知道，普通话可不是那么容易就能够说的，而他的普通话竟然如此标准，就连教六年级的老师普通话水平也不过如此。而小的那个学生竟也丝毫不惧，用同样标准，甚至其中还夹杂着英语的普通话说，\"nice，那又怎么样呢？\"这句话英语一出震惊全场，竟然有人能够将英语夹杂在普通话之中，还用的毫无违和感，就算整个李村也没有人有这样的实力，此子年纪轻轻就有如此实力，等他成长起来哪还得了？恐怕连村长都要拉拢与他。而且，等到此子成长起来，怕是会威胁村里唯一有高中生的慕容家。——虽被这句英语给震撼到，但是大的那个学生还是故作镇定：\"白话版的《三国演义》跟《水浒》，够了吗？\"六年级的很自信，所以他才亮出底牌，这两本书是他在一个破旧的书摊上买来的，尽管他日日阅读，还是不解其中的意思，但一看就是高品阶的书籍。虽然那三年级露出的微笑让他很不爽，但比赛还是开始了。六年级的率先出手，\"苹果怎么说？\"\"apple。\"\"香蕉。\"\" banana\"\"……\"一番对战下来，两人皆已负伤，三年级的撑住桌子，最后发出攻势\"西瓜用英语怎么说呢？\"六年级的慌了，西瓜，他在学这个单词的时候请假上厕所了，所以没有学会，他是怎么知道的？六年级的瘫倒在地，好像被抽光了全身力气，三年级的那个继续暴击，\"Are you okay？\"六年级的彻底落败，但他趁着三年级的那个转身之时，趁着所有人不注意之时，向三年级的那个发起了攻击\"shit。\"但被三年级的轻松躲过。等他看向周围，才发现周围的人用惊恐的眼神的看着，他急忙辩解，说，\"我不是，我没有，你们听我解释。\"\"你不要再说了，你竟然私修魔道！\"'
    }
    response = requests.post('http://localhost:8080/process_strings', json=data)
    parts, picture_descriptions=response.json().get('parts'),response.json().get('picture_descriptions')
    return parts, picture_descriptions
class solve():
    def __init__(self) -> None:
        self.control_net = ControlnetRequest()
    def getPic(self,prompt,Picnum):
        self.control_net.prompt=prompt
        print(Picnum)
        if Picnum==0:
            self.control_net.build_body(0)
            output = self.control_net.send_request()
            result = output['images'][0]
            image = Image.open(io.BytesIO(base64.b64decode(result)))
            image.save(f"/home/jzh/ai/Data/Pic/base_0.png")
            self.control_net.img_path[0]=f"/home/jzh/ai/Data/Pic/base_0.png"
        else:
            self.control_net.build_body(0)
            output = self.control_net.send_request()
            result = output['images'][0]
            image = Image.open(io.BytesIO(base64.b64decode(result)))
            image.save(f"/home/jzh/ai/Data/Pic/Canny_{Picnum}.png")
            self.control_net.img_path[1]=f"/home/jzh/ai/Data/Pic/Canny_{Picnum}.png"
            self.control_net.build_body(1)
            output = self.control_net.send_request()
            result = output['images'][0]
            image = Image.open(io.BytesIO(base64.b64decode(result)))
            image.save(f"/home/jzh/ai/Data/Pic/base_{Picnum}.png")
# def getVoice():
def getVoice(text,filename):
    data = {
    'text': text,
    'filename': filename    
    }
    response = requests.post('http://localhost:8700/process_voice', json=data)

    
 # 在这里处理输入，并生成文件
        # parts, picture_descriptions=getText()
    # s=solve()
    # parts=['一个三年级的学生和六年级的学生正在对峙，两人各自守着自己的课桌，周围的人纷纷惊叹于他们的能力。', '高的那个学生负手而立，面露不屑，用标准的普通话挑战六年级的学生。', '小的那个学生毫不畏惧，用标准的普通话和英语回击六年级的学生。', '大的那个学生故作镇定，亮出底牌，展示他珍藏的书籍。', '比赛开始，六年级的学生率先出手，向三年级的学生提问。']
    # image_text=['Two primary school students are facing each other, each defending their own desks, with people around them gaping at their abilities.', 'A tall primary school student stood there, with a disdainful expression, challenging the other person with standard Mandarin.', 'A little boy fearlessly replied with fluent English, leaving the whole audience stunned.', 'A sixth-grade student confidently shows off his best cards, displaying his treasured books.', 'The competition has started, and the sixth-grade students have taken the lead, asking the questions.']
    # for i in range(len(image_text)):
    #     if i==0:  
    #         s.getPic(image_text[i],i)
    #     else:
    #         s.getPic(image_text[0]+","+image_text[i],i)
    # for i in range(len(parts)):
    #     print(parts[i])
    #     getVoice(parts[i],f"/home/jzh/ai/Data/Voice/base_{i}.wav")
    
# 图片和音频的路径
from flask import Flask, render_template, jsonify, send_file
# 确保以下库已正确导入
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, VideoFileClip

app = Flask(__name__)

@app.route("/")
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route("/process", methods=["POST"])
def process():
    try:
         # parts, picture_descriptions=getText()
        # s=solve()
        parts=['一个三年级的学生和六年级的学生正在对峙，两人各自守着自己的课桌，周围的人纷纷惊叹于他们的能力。', '高的那个学生负手而立，面露不屑，用标准的普通话挑战六年级的学生。', '小的那个学生毫不畏惧，用标准的普通话和英语回击六年级的学生。', '大的那个学生故作镇定，亮出底牌，展示他珍藏的书籍。', '比赛开始，六年级的学生率先出手，向三年级的学生提问。']
        # image_text=['Two primary school students are facing each other, each defending their own desks, with people around them gaping at their abilities.', 'A tall primary school student stood there, with a disdainful expression, challenging the other person with standard Mandarin.', 'A little boy fearlessly replied with fluent English, leaving the whole audience stunned.', 'A sixth-grade student confidently shows off his best cards, displaying his treasured books.', 'The competition has started, and the sixth-grade students have taken the lead, asking the questions.']
        # for i in range(len(image_text)):
        #     if i==0:  
        #         s.getPic(image_text[i],i)
        #     else:
        #         s.getPic(image_text[0]+","+image_text[i],i)
        for i in range(len(parts)):
            print(parts[i])
            getVoice(parts[i],f"/home/jzh/ai/Data/Voice/base_{i}.wav")
    
        video_files = []
        # 循环中的处置步骤猜想你有可用的音频和图片资源数目相等
        for i in range(5):  # 假设你有5对音频和图片
            image_file = f'/home/jzh/ai/Data/Pic/base_{i}.png'
            audio_file = f'/home/jzh/ai/Data/Voice/base_{i}.wav'
            output_video = f'/home/jzh/ai/Data/Output/base_{i}.mp4'
            
            # 创建图片clip
            print(image_file)
            image_clip = ImageClip(image_file)
            # 创建音频clip
            audio_clip = AudioFileClip(audio_file)
            # 设置图片clip的时长与音频一样
            print(123)
            image_clip = image_clip.set_duration(audio_clip.duration)
            
            # 创建视频文件
            video = image_clip.set_audio(audio_clip)
            video.write_videofile(output_video, fps=24)
            video_files.append(output_video)
            
            # 关闭clip资源
            audio_clip.close()
            image_clip.close()

        # 合并视频文件
        clips = [VideoFileClip(video) for video in video_files]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile("/home/jzh/ai/Data/Output/output.mp4")
        
        # 关闭所有clip资源
        [clip.close() for clip in clips]
        final_clip.close()
        
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route("/download")
def download():
    try:
        return send_file("/home/jzh/ai/Data/Output/output.mp4", as_attachment=True)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)