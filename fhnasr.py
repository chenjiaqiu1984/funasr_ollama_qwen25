

import os
import time



import datetime
from openai import OpenAI
from funasr import AutoModel

#这个函数用于与qwen模型通信
def chat_with_ollama(messages: list[dict],modelname):
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    #电脑比较懒，用的7b模型
    model = os.getenv('OLLAMA_MODEL', modelname)


    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv('OLLAMA_API_KEY', 'ollama')
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content




#这个函数用来转换文件
#参数textpath没啥用了，原来是输出.txt文件
def convert_media_to_text(media_path,text_path,model,md_pat):
    audio_path = media_path



    res = model.generate(input=audio_path,
                         batch_size_s=300,
                         hotword='魔搭')
    #print(type(res),res)
    text=res[0]["text"]
    #print(text)


    print(datetime.datetime.now(),"convert_media_to_text ",media_path, text_path,audio_path)
    modlename1="wangshenzhi/llama3-8b-chinese-chat-ollama-q8"
    modlename2 = "llama3.1"
    modlename3 = "llama2-chinese"
    modlename4 = "wangshenzhi/llama3-8b-chinese-chat-ollama-fp16"
    modlename1 = "qwen2.5:7b"
    modlename1 = "deepseek-r1:8b"
    
    modlename6 = "gemma2"
    
    content = text
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(content)

    messages1 = [
        {"role": "system",
         "content": "提供的文案进行仔细校对，纠正错别字，添加或调整标点符号，以提高文案的清晰度和专业性。"},
        {"role": "user", "content": content},  # 无实际意义，用于补充缺失的user消息
    ]

    messages2 = [
        {"role": "system",
         "content": "进行内容提炼与总结，并为其打上恰当的标签。提炼的内容需包括：视频标题、标签、一句话总结文章、详细文章大纲（摘要形式），以及基于大纲的完整文章要点总结摘要。阅读并理解信息：仔细阅读输入的内容，确保对其主题、核心观点和细节有全面理解。根据输入内容的核心主题，构思一个简洁明了、吸引人的视频标题。根据输入内容所属的领域、学科或行业类别，为其打上2-3个最相关的标签。用一句话概括输入内容的核心观点或主旨。摘要形式列出文章的主要部分，每个部分用简短语句描述其核心内容。确保大纲能够完整体现文章的要点，并且结构清晰。大纲应包含引言、主体内容、结论等部分。基于上述大纲，进一步细化每个部分的要点，生成一个更加详细的摘要。摘要应包含文章中的关键信息、数据、观点等，确保读者能够通过摘要快速了解文章的主要内容。"},
        {"role": "user", "content": content},  # 无实际意义，用于补充缺失的user消息
    ]

    messages3 = [
        {"role": "system",
         "content": "智能扩写文章,使整体文章内容自然流畅。扩写的文本内容，要求语言生动，贴合日常生活风格，不要文字刻板，不要有很重的AI味道。避免使用'首先、其次、再有、总而言之'等机械性的总结语句。"},
        {"role": "user", "content": content},  # 无实际意义，用于补充缺失的user消息
    ]



    mddocument1 = chat_with_ollama(messages1,modlename1)
    mddocument2 = chat_with_ollama(messages2,modlename1)
    mddocument3 = chat_with_ollama(messages3,modlename1)


    #mddocument5 = chat_with_ollama(messages1, modlename1)
    #mddocument6 = chat_with_ollama(messages2, modlename1)
    #mddocument7 = chat_with_ollama(messages3, modlename1)
    #mddocument8 = chat_with_ollama(messages4, modlename1)

    with open(md_pat,"w", encoding="utf-8") as f:
        f.write("\n\n\n##################""录音转换文字#############################\n\n\n")
        f.write(content)
        #modlename1 = "wangshenzhi/llama3-8b-chinese-chat-ollama-q8"
        #modlename1 = "wangshenzhi/llama3-8b-chinese-chat-ollama-fp16"
        f.write("\n\n\n##################"+modlename1+"#############################\n\n\n")
        f.write("\n\n\n##################"+modlename1+"段落标题#############################\n\n\n")
        f.write(mddocument1)
        f.write("\n\n\n##################"+modlename1+"概要思路#############################\n\n\n")
        f.write(mddocument2)
        f.write("\n\n\n##################"+modlename1+"直播脚本#############################\n\n\n")
        f.write(mddocument3)









def process_directory(directory,model):
    for root,_,files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp3','.wav','.mp4','.avi', '.mkv','.m4a','.flv','.mov')):
            #if file.endswith('.flv'):
                media_path=os.path.join(root,file)

                text_path=os.path.join(root,file.rsplit('.',1)[0]+'.txt')
                md_Pat=os.path.join(root,file.rsplit('.',1)[0]+'.md')

                #已经有txt文件的删掉，这段可以删除，因为我原来已经跑过txt的脚本，想先把不用的格式删掉
                #if(os.path.exists(text_path)):
                    #os.remove(text_path)

                #print(datetime.datetime.now(),"process_directory",directory,media_path, text_path)

                #如果以前生成过文件，先删除
                if(os.path.exists(text_path)):
                    #os.remove(md_Pat)
                    print("skip")

                else:
                    print("not remove md")
                    convert_media_to_text(media_path, text_path, model, md_Pat)



                    #转换的主执行





if __name__ == "__main__":


    directory = "F:/005.育儿"  # 替换为你的目录路径
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(model="paraformer-zh",
                      vad_model="fsmn-vad",
                      punc_model="ct-punc-c",
                      device="cuda"  #没装cuda可以注释掉
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )

    process_directory(directory,model)