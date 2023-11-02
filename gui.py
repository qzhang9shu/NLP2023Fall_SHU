# keyword_gui.py

import tkinter as tk
from tkinter import scrolledtext, IntVar,messagebox
from extract import extract_combined_keywords_from_text


def is_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def display_keywords():
    text = text_input.get("1.0", tk.END)

    if not is_chinese(text):
        messagebox.showwarning("警告", "请输入中文文本!")
        return

    num_keywords = num_keywords_var.get()
    model_keywords, tfidf_keywords, combined_keywords = extract_combined_keywords_from_text(text,
                                                                                            num_keywords=num_keywords)

    model_keywords_output.delete("1.0", tk.END)
    model_keywords_output.insert(tk.END, ', '.join(model_keywords))

    tfidf_keywords_output.delete("1.0", tk.END)
    tfidf_keywords_output.insert(tk.END, ', '.join(tfidf_keywords))

    combined_keywords_output.delete("1.0", tk.END)
    combined_keywords_output.insert(tk.END, ', '.join(combined_keywords))
# 创建主窗口
root = tk.Tk()
root.title("关键词提取")

# 创建输入框
label = tk.Label(root, text="请输入中文文本:")
label.pack(pady=10)
text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
text_input.pack(pady=10)

# 创建关键词数量输入框
label_num_keywords = tk.Label(root, text="TF-IDF关键词数量:")
label_num_keywords.pack(pady=10)
num_keywords_var = IntVar(value=8)  # 默认值为8
num_keywords_entry = tk.Entry(root, textvariable=num_keywords_var)
num_keywords_entry.pack(pady=10)

# 创建按钮
button = tk.Button(root, text="提取关键词", command=display_keywords)
button.pack(pady=10)

# 创建输出框
label1 = tk.Label(root, text="LSTM预测关键词:")
label1.pack(pady=10)
model_keywords_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5)
model_keywords_output.pack(pady=10)

label2 = tk.Label(root, text="TF-IDF预测关键词:")
label2.pack(pady=10)
tfidf_keywords_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5)
tfidf_keywords_output.pack(pady=10)

label3 = tk.Label(root, text="合并关键词:")
label3.pack(pady=10)
combined_keywords_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5)
combined_keywords_output.pack(pady=10)

root.mainloop()