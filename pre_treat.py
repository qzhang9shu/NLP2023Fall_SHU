import pandas as pd
import jieba
def all_in(data):
    counts=0
    whether=[]
    # 遍历每一行数据
    for index, row in data.iterrows():
        text = row['text'].split()
        keywords = row['keywords'].split('_')
        # 检查每个关键词是否都在文本中
        all_present = all(keyword in text for keyword in keywords)
        if all_present:
            counts=counts+1
        whether.append(all_present)
    # 显示检查结果
    data_filter=data[whether]
    data.reset_index(drop=True)
    return data_filter

def contain_English(data):
    mask=data['text'].str.contains('[a-zA-Z]',regex=True)
    data_filter=data[~mask]
    return data_filter

def words_num5(data):
    whether = []
    # 遍历每一行数据
    for index, row in data.iterrows():
        text = row['text']
        keywords = row['keywords'].split('_')
        # print(text,keywords)
        # 检查每个关键词是否都在文本中
        within_5 = len(keywords)<10 and len(keywords)>1
        whether.append(within_5)
    # 显示检查结果
    data_filter = data[whether]
    return data_filter

def pre_treat(data):
    # data=pd.read_csv('train.tsv',sep='\t',names=['1','text','keywords'])
    data=data.iloc[:,1:]
    # data_finish=words_num5(contain_English(all_in(data)))
    # data_finish_split=data_finish
    # data_finish_split['text'] = data_finish_split['text'].apply(lambda x: " ".join(jieba.cut(x)))
    # data_finish_split.to_csv("data_train_split.tsv",index=False)
    # data.to_csv("data_test.tsv",index=False)


data=pd.read_csv('alldata3.tsv')
# data=pd.read_csv('alldata3.tsv',sep='\t',names=['3','text','keywords','2','1'],on_bad_lines='skip')
# data = data.iloc[:, 1:3]
# print(data)



data['text'] = data['text'].apply(lambda x: " ".join(jieba.cut(x)))
data_finish=words_num5(all_in(data))
data_finish_split=data_finish
data_finish_split.to_csv("alldata2.tsv",index=False)
data_train= data_finish_split.iloc[0:int(9*len(data_finish_split)/10),:]
data_test= data_finish_split.iloc[int(9*len(data_finish_split)/10):,:]
data_train.to_csv("data_train2.tsv",index=False)
data_test.to_csv("data_test2.tsv",index=False)
