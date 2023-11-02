from pyhanlp import *


def segment_paragraphs(input_file, output_file):
    # 加载HanLP的分词器
    CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")

    # 打开输入和输出文件
    with open(input_file, 'r', encoding='utf-8') as f_input, open(output_file, 'w', encoding='utf-8') as f_output:
        # 逐行读取输入文件
        for line in f_input:
            # 使用空行进行段落分隔
            paragraph = line.strip()
            if not paragraph:
                continue

            # 分词并写入输出文件
            # seg_result = tokenizer.segment(paragraph)
            segment = HanLP.newSegment().enableCustomDictionary(True)
            term_list = segment.seg(paragraph)
            word_list = [str(term.word) for term in term_list]
            output_line = ' '.join(word_list)
            f_output.write(output_line + '\n')

    print("分词完成！")


# 调用函数进行分词并保存结果
input_file = "train_400.txt"
output_file = "train_400_output.txt"
segment_paragraphs(input_file, output_file)