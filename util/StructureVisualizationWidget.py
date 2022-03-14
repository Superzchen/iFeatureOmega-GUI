from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
import sys

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('显示网页')
        self.resize(800, 800)
        layout = QVBoxLayout(self)
        # 新建一个QWebEngineView()对象
        self.qwebengine = QWebEngineView()
        layout.addWidget(self.qwebengine)
        # 设置网页在窗口中显示的位置和大小
        # 在QWebEngineView中加载网址
        self.qwebengine.load(QUrl(r"https://www.csdn.net/"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())