'''
功能
	翻译软件,支持:
		-百度翻译
		-有道翻译
		-谷歌翻译
作者:
	Ahab
公众号:
	Ahab杂货铺
User-Agent和Cookie 需要自行添加
'''
from predict import pred,init
import re
import js
import sys
import time
import js2py
import random
import hashlib
import requests
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLabel, QLineEdit, QPushButton,QTextEdit,QColorDialog, QFontDialog
from PyQt5.QtGui import QPainter, QColor, QFont
infer_model,infer_sess = init()
class Demo(QWidget):
	def __init__(self, parent=None):
		super().__init__()
		Label = QLabel(self)
		self.resize(1000, 800)
		Label.setStyleSheet("QLabel{border-image: url(./beijing.jpg);}")
		Label.setFixedWidth(1000)
		Label.setFixedHeight(800)
		self.setFont(QFont("微软雅黑",20,QFont.Bold))
		self.setWindowTitle('新闻提取')
		self.Label1 = QLabel('原文')
		self.Label2 = QLabel('提取')
		self.LineEdit1 = QTextEdit()
		self.LineEdit2 = QLineEdit()
		self.translateButton = QPushButton()
		self.translateButton.setText('运行')
		self.grid = QGridLayout()
		self.grid.setSpacing(20)
		self.grid.addWidget(self.Label1, 1, 0)
		self.grid.addWidget(self.LineEdit1,1, 1)
		self.grid.addWidget(self.Label2, 2, 0)
		self.grid.addWidget(self.LineEdit2, 2, 1)
		self.grid.addWidget(self.translateButton, 2, 2)
		self.setLayout(self.grid)
		self.translateButton.clicked.connect(lambda : self.translate())

	def translate(self):
		word = self.LineEdit1.toPlainText()
		if not word:
			return
		results = pred(source_str = word,infer_model=infer_model,infer_sess=infer_sess)
		print(results)
		self.LineEdit2.setText(results)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	demo = Demo()
	demo.show()

sys.exit(app.exec_())