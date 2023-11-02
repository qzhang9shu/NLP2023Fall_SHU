"""
作者: lingengyuan
时间: 2023年 10月 19日
"""


class DFAFilter:
    def __init__(self):
        self.keyword_tree = {}
        self.keyword_set = set()
        self.replace_char = '*'

    def add_keyword(self, keyword):
        node = self.keyword_tree
        for char in keyword:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["_end"] = "_end"

    def build(self):
        queue = []
        for key in self.keyword_tree:
            queue.append((self.keyword_tree[key], key))
        while queue:
            node, word = queue.pop(0)
            if "_end" in node:
                self.keyword_set.add(word)
            for char in node:
                if char != "_end":
                    new_word = word + char
                    queue.append((node[char], new_word))

    def load_keywords_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                keyword = line.strip()
                if keyword:
                    self.add_keyword(keyword)

    def filter(self, text):
        result_text = list(text)
        i = 0
        while i < len(text):
            node = self.keyword_tree
            j = i
            while j < len(text) and text[j] in node:
                node = node[text[j]]
                j += 1
            if "_end" in node:
                for k in range(i, j):
                    result_text[k] = self.replace_char
                i = j
            else:
                i += 1
        return ''.join(result_text)


if __name__ == "__main__":
    dfa_filter = DFAFilter()

    # 从文本文件加载敏感词库
    dfa_filter.load_keywords_from_file("sensitive_words.txt")

    # 构建DFA树
    dfa_filter.build()

    text = "你是不是喜欢藏独和胡锦涛"

    filtered_text = dfa_filter.filter(text)
    print("替换后的文本:", filtered_text)
