# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

'''
首页
'''

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit
from PyQt5.QtCore import QCoreApplication

import codecs
from preprocess import *
from TF_IDF import *
import Error_GUI

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(650, 605)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_mainwindow_commit1 = QtWidgets.QPushButton(self.centralwidget)
        self.button_mainwindow_commit1.setGeometry(QtCore.QRect(180, 520, 80, 34))
        self.button_mainwindow_commit1.setObjectName("button_mainwindow_commit1")
        self.label_mainwindow_text = QtWidgets.QLabel(self.centralwidget)
        self.label_mainwindow_text.setGeometry(QtCore.QRect(80, 140, 60, 16))
        self.label_mainwindow_text.setObjectName("label_mainwindow_text")
        self.label_mainwindow_result1 = QtWidgets.QLabel(self.centralwidget)
        self.label_mainwindow_result1.setGeometry(QtCore.QRect(80, 300, 71, 16))
        self.label_mainwindow_result1.setObjectName("label_mainwindow_result1")
        self.label_mainwindow_welcome = QtWidgets.QLabel(self.centralwidget)
        self.label_mainwindow_welcome.setGeometry(QtCore.QRect(180, 80, 321, 16))
        self.label_mainwindow_welcome.setObjectName("label_mainwindow_welcome")
        self.label_mainwindow_title = QtWidgets.QLabel(self.centralwidget)
        self.label_mainwindow_title.setGeometry(QtCore.QRect(195, 20, 260, 40))
        self.label_mainwindow_title.setMinimumSize(QtCore.QSize(0, 0))
        self.label_mainwindow_title.setMaximumSize(QtCore.QSize(500, 300))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(28)
        self.label_mainwindow_title.setFont(font)
        self.label_mainwindow_title.setObjectName("label_mainwindow_title")
        self.button_mainwindow_commit2 = QtWidgets.QPushButton(self.centralwidget)
        self.button_mainwindow_commit2.setGeometry(QtCore.QRect(370, 520, 80, 34))
        self.button_mainwindow_commit2.setObjectName("button_mainwindow_commit2")
        self.label_mainwindow_result2 = QtWidgets.QLabel(self.centralwidget)
        self.label_mainwindow_result2.setGeometry(QtCore.QRect(80, 440, 71, 16))
        self.label_mainwindow_result2.setObjectName("label_mainwindow_result2")
        self.textEdit_1 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_1.setGeometry(QtCore.QRect(180, 120, 420, 140))
        self.textEdit_1.setObjectName("textEdit_1")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(180, 280, 420, 140))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(180, 440, 201, 34))
        self.textEdit_3.setObjectName("textEdit_3")
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 650, 22))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

        # 点击button_mainwindow_commit1后进行button_click函数
        self.button_mainwindow_commit1.clicked.connect(self.button_click_mainwindow_commit1)

        # 点击button_mainwindow_commit2后进行button_click函数
        self.button_mainwindow_commit2.clicked.connect(self.button_click_mainwindow_commit2)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "关键词处理系统"))
        self.button_mainwindow_commit1.setText(_translate("mainWindow", "预处理"))
        self.label_mainwindow_text.setText(_translate("mainWindow", "文本内容："))
        self.label_mainwindow_result1.setText(_translate("mainWindow", "预处理结果："))
        self.label_mainwindow_welcome.setText(
            _translate("mainWindow", "欢迎您登录此系统，请在下方文本框内填写相应的文本内容。"))
        self.label_mainwindow_title.setText(_translate("mainWindow", "关键词提取系统"))
        self.button_mainwindow_commit2.setText(_translate("mainWindow", "结果"))
        self.label_mainwindow_result2.setText(_translate("mainWindow", "关键词："))
        self.textEdit_1.setHtml(_translate("mainWindow",
                                           "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                           "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                           "p, li { white-space: pre-wrap; }\n"
                                           "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                           "<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textEdit_2.setHtml(_translate("mainWindow",
                                           "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                           "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                           "p, li { white-space: pre-wrap; }\n"
                                           "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                           "<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textEdit_3.setHtml(_translate("mainWindow",
                                           "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                           "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                           "p, li { white-space: pre-wrap; }\n"
                                           "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                           "<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))

    def button_click_mainwindow_commit1(self):
        # 获取textEdit中的文本内容
        gettext = self.textEdit_1.toPlainText()
        # self.lineEdit_mainwindow_result1.setText(gettext)

        list_preprocessed = text_preprocess(gettext)  # 数据预处理

        # 打开文件，以写模式写入
        with codecs.open('./test/test.txt', 'w', 'utf-8') as file:
            file.write(' '.join(list_preprocessed))

        # 将list转变成str格式，储存到content_test中，方便显示
        list_preprocessed.append('')  # 给list的尾部+一个空元素

        i = 0
        j = 0
        # print(list_removed)
        content_test = ''  # 用于在textEdit_2上展示预处理结果
        while list_preprocessed[i] != '':
            # print(list_removed[i])
            content_test = content_test + list_preprocessed[i] + ' '

            # 判断是否存在非中文文本的输入
            if (list_preprocessed[i] > '\u9fff') or (list_preprocessed[i] < '\u4e00'):
                j = 1

            i = i + 1

        # 如果存在非中文文本的输入
        if j == 1:
            # 显示提示窗口
            self.new = Error_GUI.Ui_MainWindow()
            self.new.show()

        # print(content_test)

        self.textEdit_2.setText(content_test)  # 在textEdit_2中显示content_test

        text_cleaned = clean_text(gettext)
        segment = HanLP.newSegment().enableCustomDictionary(True)
        term_list = segment.seg(text_cleaned)  # 分词结果

        term_list1 = []
        for number in range(len(term_list)):
            # print(term_list[number])
            term_list[number] = str(term_list[number])
            # print(type(term_list[number]))
            term_list1.append(term_list[number])

        # 打开文件，以写模式写入
        with codecs.open('./test/test1.txt', 'w', 'utf-8') as file:
            file.write(' '.join(term_list1))


    def button_click_mainwindow_commit2(self):
        # # 获取textEdit中的文本内容
        # gettext = self.textEdit_1.toPlainText()
        # # self.lineEdit_mainwindow_result1.setText(gettext)
        #
        # list_preprocessed = text_preprocess(gettext)  # 数据预处理
        #
        # # 将list转变成str格式，储存到content_test中，方便显示
        # list_preprocessed.append('')  # 给list的尾部+一个空元素
        #
        # i = 0
        # j = 0
        # # print(list_removed)
        # content_test = ''  # 用于在textEdit_2上展示预处理结果
        # while list_preprocessed[i] != '':
        #     # print(list_removed[i])
        #     content_test = content_test + list_preprocessed[i] + ' '
        #
        #     # 判断是否存在非中文文本的输入
        #     if (list_preprocessed[i] > '\u9fff') or (list_preprocessed[i] < '\u4e00'):
        #         j = 1
        #
        #     i = i + 1
        #
        # # 如果存在非中文文本的输入
        # if j == 1:
        #     # 显示提示窗口
        #     self.new = Error_GUI.Ui_MainWindow()
        #     self.new.show()
        #
        # # print(content_test)
        #
        # self.textEdit_2.setText(content_test)  # 在textEdit_2中显示content_test

        doc_list_train_src = load_data(corpus_path='./data/train_src.txt')
        doc_list_test_src = load_data(corpus_path='./test/test.txt')

        # text_cleaned = clean_text(gettext)
        # segment = HanLP.newSegment().enableCustomDictionary(True)
        # term_list = segment.seg(text_cleaned)  # 分词结果
        #
        # term_list1 = []
        # for number in range(len(term_list)):
        #     # print(term_list[number])
        #     term_list[number] = str(term_list[number])
        #     # print(type(term_list[number]))
        #     term_list1.append(term_list[number])
        #
        # # 打开文件，以写模式写入
        # with codecs.open('./test/test1.txt', 'w', 'utf-8') as file:
        #     file.write(' '.join(term_list1))

        keywords_list = load_data(corpus_path='./test/test1.txt')
        # print(doc_list_test_src)

        TFIDF(savepath='./result/test.txt',  # TF-IDF结果保存路径
              keyword_num=5,  # 候选词个数
              end=0,  # end代表运行到第几行，设置为0时表示全部
              save=True,  # 是否保存结果
              doc_list_train_src=doc_list_train_src,
              doc_list_test_src=doc_list_test_src,
              keywords_list=keywords_list)

        # 打开文件
        with open('./result/test.txt', 'r', encoding='utf-8') as file:
            # 读取文件的全部内容
            # print(1)
            content = file.read()
            # print(2)

        # 打印文件的全部内容
        # print(content)

        content = content.replace('预测值', '').replace('\n', '').replace('{', '').replace('}', '').replace('\'',
                                                                                                            '').replace(
            ',', '')

        # print(content)
        self.textEdit_3.setText(content)  # 在textEdit_3中显示content


