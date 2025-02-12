from pdf2docx import Converter
import os


def convert_word(pdf_path, text_path):
    cv = Converter(pdf_path)
    cv.convert(text_path)
    cv.close()
def process_directory(directory,model):
    if model=='word':
        for root,_,files in os.walk(directory):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path=os.path.join(root,file)
                    text_path=os.path.join(root,file.rsplit('.',1)[0]+'.docx')
                    if(os.path.exists(text_path)):
                        print("skip")
                    else:
                        print("process to word")
                        convert_word(pdf_path, text_path)
if __name__ == "__main__":
    directory = "C:/Users/Administrator/Desktop/timu"  # 替换为你的目录路径
    process_directory(directory,"word")




