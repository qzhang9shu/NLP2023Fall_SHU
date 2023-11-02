'''import speech_recognition as sr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dfa_filter import DFAFilter

# 加载模型和分词器
model_path = 'mymodel.pth'  # 模型文件的路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

while True:
    # 选择输入方式
    input_mode = input("请选择输入方式：\n1. 语音输入\n2. 打字输入\n3. 退出\n输入选项（1、2或3）：")

    if input_mode == "1":
        # 使用语音识别库进行语音转文本
        r = sr.Recognizer()
        mic = sr.Microphone()

        # 使用麦克风录音
        with mic as source:
            print("请开始说话...")
            audio = r.listen(source)

        # 停止录音，获取部分识别结果
        print("识别中...")
        try:
            text_to_predict = r.recognize_google(audio, language='zh-CN', show_all=True)
            if text_to_predict:
                # 提取最后一段识别结果
                final_result = text_to_predict['alternative'][-1]['transcript']
                print("转录结果:", final_result)
            else:
                print("未能识别到语音。请重新尝试。")
                continue
        except sr.UnknownValueError:
            print("无法识别语音。请重新尝试。")
            continue
        except sr.RequestError as e:
            print("无法连接到语音识别服务。请检查网络连接。")
            continue

    elif input_mode == "2":
        text_to_predict = input("请输入文本：")
        # 停止录音，获取部分识别结果
        print("识别中...")

    elif input_mode == "3":
        print("程序已退出。")
        break
    else:
        print("无效的选项。请重新选择正确的选项。")
        continue

    # 对输入文本进行预处理
    inputs = tokenizer(final_result, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    dfa_filter = DFAFilter()

    # 从文本文件加载敏感词库
    dfa_filter.load_keywords_from_file("keywords.txt")

    # 构建DFA树
    dfa_filter.build()

    # 输出预测结果
    if predicted_label == 0:
        print("无攻击性")
    else:
        print("攻击性")

    filtered_text = dfa_filter.filter(final_result)
    print("替换后的文本:", filtered_text)'''

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import speech_recognition as sr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dfa_filter import DFAFilter

# 加载模型和分词器
model_path = 'mymodel.pth'  # 模型文件的路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 创建DFA过滤器
dfa_filter = DFAFilter()

# 从文本文件加载敏感词库
dfa_filter.load_keywords_from_file("keywords.txt")

# 构建DFA树
dfa_filter.build()

# 创建Tkinter窗口
window = tk.Tk()
window.title("语音/文本攻击性检测")
window.geometry("600x550")
window.configure(bg='#2b2b2b')  # 设置背景颜色为深色

# 创建样式
style = ttk.Style()
style.theme_use('clam')  # 使用clam主题
style.configure('.', background='#2b2b2b', foreground='white')  # 设置前景和背景颜色
style.configure('TButton', foreground='black')  # 设置按钮文本颜色为黑色
style.configure('TLabel', foreground='black')  # 设置标签文本颜色为黑色

# 创建标签和文本框
label1 = ttk.Label(window, text="请选择输入方式：", font=("Arial", 12), foreground="white")
label1.pack(pady=10)

label2 = ttk.Label(window, text="1. 语音输入\n2. 打字输入\n3. 退出", font=("Arial", 12), foreground="white")
label2.pack()

entry = ttk.Entry(window, font=("Arial", 12), foreground="black")
entry.pack(pady=10)

text1 = tk.Text(window, height=10, font=("Arial", 12))
text1.pack(pady=10)

label3 = ttk.Label(window, text="请在上面的文本框中输入文本，然后按确定按钮", font=("Arial", 12), foreground="white")
label3.pack()

text2 = tk.Text(window, height=10, font=("Arial", 12))
text2.pack(pady=10)

scrollbar = ttk.Scrollbar(window)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text2.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=text2.yview)


# 定义语音识别函数
def recognize_speech():
    text2.delete('1.0', tk.END)  # 清空结果文本框内容
    r = sr.Recognizer()
    mic = sr.Microphone()

    # 使用麦克风录音
    with mic as source:
        messagebox.showinfo("提示", "请开始说话...")
        audio = r.listen(source)

    # 停止录音，获取部分识别结果
    messagebox.showinfo("提示", "识别中...")
    try:
        text_to_predict = r.recognize_google(audio, language='zh-CN', show_all=True)
        if text_to_predict:
            # 提取最后一段识别结果
            final_result = text_to_predict['alternative'][-1]['transcript']
            text2.insert(tk.END, "转录结果: " + final_result + "\n")
            predict_text(final_result)  # 调用预测函数
        else:
            text2.insert(tk.END, "未能识别到语音。请重新尝试。\n")
    except sr.UnknownValueError:
        text2.insert(tk.END, "无法识别语音。请重新尝试。\n")
    except sr.RequestError as e:
        text2.insert(tk.END, "无法连接到语音识别服务。请检查网络连接。\n")


# 定义预测函数
def predict_text(text):
    # 提取特征
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    offensive_probability = probabilities[1] * 100  # 获取攻击性类别的概率

    # 判断是否具有攻击性
    is_offensive = offensive_probability >= 0.5

    # 过滤敏感词汇
    filtered_text = dfa_filter.filter(text)
    # 显示结果
    text2.insert(tk.END, f"过滤后文本: {filtered_text}\n")
    if is_offensive:
        text2.insert(tk.END, "具有攻击性\n")
    else:
        text2.insert(tk.END, "不具有攻击性\n")
    text2.insert(tk.END, f"攻击性概率: {offensive_probability:.2f}%\n")




# 定义按钮点击事件
def button_click():
    text2.delete('1.0', tk.END)  # 清空结果文本框内容
    option = entry.get()

    if option == '1':
        recognize_speech()
    elif option == '2':
        text = text1.get('1.0', tk.END).strip()
        if text:
            predict_text(text)
        else:
            text2.insert(tk.END, "请输入文本后再点击确定按钮。\n")
    elif option == '3':
        window.destroy()
    else:
        text2.insert(tk.END, "无效的选项。请重新输入。\n")


# 创建按钮
button = ttk.Button(window, text="确定", command=button_click)
button.pack(pady=10)

# 运行窗口
window.mainloop()