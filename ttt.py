# 打开文件
with open('./result/test.txt', 'r', encoding='utf-8') as file:
    # 读取文件的全部内容
    content = file.read()

file.close()
# 打印文件的全部内容
print(content)
content = content.replace('预测值', '').replace('\n', '').replace('{', '').replace('}', '').replace('\'', '').replace(',', '')
print(content)
