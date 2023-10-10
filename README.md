# 上海大学2023学年秋季NLP课程项目
本项目为**图片描述生成(Image Captioning)**，旨在通过机器学习，让计算机对指定图像生成描述性语句。

## 项目概述
### 1. 编码器与解码器结构
* 编码器：采用卷积神经网络(CNN)，以 **ResNet-101** 作为预训练模型，实现对图像的编码，决定了整个模型的图像识别能力。
* 解码器：采用循环神经网络(RNN)，主要功能是读取编码后的图像并生成文本描述。

### 2. 注意力机制
为了解决编码器-解码器在处理固定长度向量时的局限性，在此引入注意力机制。注意力机制通过增加一个上下文向量来对每个时间步的输入进行解码，以增强图像区域和单词的相关性，从而获取更多的图像语义细节。

### 3. Beam Search
对贪婪搜索的一个改进，可以令解码器在每个解码步骤中选择得分最高的单词，使结果更准确。

## 环境配置
以下是我运行此项目的环境：
* Python 3.7
* CUDA 11.6
* cudnn 8.6
* pytorch 1.13.1
* Others (See ```requirements.txt```)

## 实施步骤
1. 下载coco2017数据集
2. 运行train.py进行训练
3. 运行eval.py进行评估
4. 运行caption.py进行推理

## 文件结构：
```
  ├── data: 数据集根目录
	  ├── coco2017: coco2017数据集目录
	     ├── train2017: 所有训练图像文件夹(118287张)
	     ├── val2017: 所有验证图像文件夹(5000张)
	     ├── annotations: 对应标注文件夹
	              ├── instances_train2017.json: 对应目标检测、分割任务的训练集标注文件
	              ├── instances_val2017.json: 对应目标检测、分割任务的验证集标注文件
	              ├── captions_train2017.json: 对应图像描述的训练集标注文件
	              ├── captions_val2017.json: 对应图像描述的验证集标注文件
	              ├── person_keypoints_train2017.json: 对应人体关键点检测的训练集标注文件
	              └── person_keypoints_val2017.json: 对应人体关键点检测的验证集标注文件夹
	     └── dataset_coco.json: 根据标注文件重新整理的图像数据
  ├── results
	  ├── caption: 数据集预处理文件夹
	  ├── checkpoints: 训练模型保存文件夹
  ├── caption.py: 使用beam search生成caption并进行可视化
  ├── datasets.py: 继承pytorch的DataSet类
  ├── eval.py: 评估模型
  ├── models.py: 定义模型结构
  ├── train.py: 训练模型
  └── utils.py: 包含其它各种功能函数的辅助文件
```

## 参考资料
* [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
* [Image Captioning项目实战](https://zhuanlan.zhihu.com/p/424132486)
* [deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
