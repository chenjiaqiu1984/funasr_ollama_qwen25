from funasr import AutoModel
import os
import time

import whisper
import ffmpeg
import datetime
from openai import OpenAI



def chat_with_ollama(messages: list[dict]):
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    model = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')


    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv('OLLAMA_API_KEY', 'ollama')
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


def on_message(content):
    msg.stream_token(content)


def convert_media_to_text(media_path,text_path,model,md_pat):
    isVideo = 0
    audio_path = media_path
    print(datetime.datetime.now(), "audio file ", media_path)

    '''
    isVideo=0
    if(media_path.endswith(('.wav'))):
        isVideo = 0
        audio_path = media_path
        print(datetime.datetime.now(), "audio file ", media_path)

    else:
        isVideo = 1
        audio_path = media_path.rsplit('.', 1)[0] + '.wav'
        ffmpeg.input(media_path).output(audio_path).run()
        print(datetime.datetime.now(), "not audio file ", media_path)
    '''
    res = model.generate(input=audio_path,
                         batch_size_s=300,
                         hotword='魔搭')
    #print(type(res),res)
    text=res[0]["text"]
    #print(text)


    #print(datetime.datetime.now(),"convert_media_to_text ",media_path, text_path,audio_path)

    if isVideo==1:
        isVideo=0
        os.remove(audio_path)
        print(datetime.datetime.now(),"remove audio file ",audio_path)

    content = text

    messages = [
        {"role": "system",
         "content": "重新将这段文字整理分类，使其表达更加流畅。"},
        {"role": "user", "content": content},  # 无实际意义，用于补充缺失的user消息
    ]

    mddocument1 = chat_with_ollama(messages)



    messages = [
        {"role": "system",
         "content": "整理文章的概要与思路.用到的案例有哪些。"},
        {"role": "user", "content": content},  # 无实际意义，用于补充缺失的user消息
    ]

    mddocument2 = chat_with_ollama(messages)


    messages = [
        {"role": "system",
         "content": "你是一个资深的青少年心理咨询师，基于给出的文字，输出至少5000字以上的文稿，用于直播使用。"},
        {"role": "user", "content": content},  # 无实际意义，用于补充缺失的user消息
    ]

    mddocument3 = chat_with_ollama(messages)





    with open(md_pat,"w", encoding="utf-8") as f:
        f.write("\n\n\n##################录音转换文字#############################\n\n\n")
        f.write(text)
        f.write("\n\n\n##################段落标题#############################\n\n\n")
        f.write(mddocument1)
        f.write("\n\n\n##################概要思路#############################\n\n\n")
        f.write(mddocument2)
        f.write("\n\n\n##################直播脚本#############################\n\n\n")
        f.write(mddocument3)






def process_directory(directory,model):
    for root,_,files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp3','.wav','.mp4','.avi', '.mkv','.m4a','flv')):
            #if file.endswith('.flv'):
                media_path=os.path.join(root,file)

                text_path=os.path.join(root,file.rsplit('.',1)[0]+'.txt')
                md_Pat=os.path.join(root,file.rsplit('.',1)[0]+'.md')

                if(os.path.exists(text_path)):
                    os.remove(text_path)

                #print(datetime.datetime.now(),"process_directory",directory,media_path, text_path)

                if(os.path.exists(md_Pat)):
                    print("skip")
                else:
                    convert_media_to_text(media_path,text_path,model,md_Pat)




if __name__ == "__main__":


    directory = "F:/017完型萨提亚"  # 替换为你的目录路径
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      device="cuda"
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )

    process_directory(directory,model)