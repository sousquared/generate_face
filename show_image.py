# reference
# show images:
#  https://qiita.com/Nobu12/items/36bc4cc6cf4163b66847
#  https://symfoware.blog.fc2.com/blog-entry-2299.html
# images from np.array
#  https://qiita.com/ceptree/items/c3a7c52cdd152c9f62d9

import sys

import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


from PIL import Image


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.width = 680
        self.height = 510

        self.root = './images/'
        self.FileName = 'generator.png'

        self.initUI()


    def initUI(self):

        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle('show an image')

        # Widget for Figure
        self.FigureWidget = QWidget(self)
        # add Layout to FigureWidget
        self.FigureLayout = QVBoxLayout(self.FigureWidget)

        # set a figure
        self.Figure = plt.figure()
        # add Figure to FigureCanvas
        self.FigureCanvas = FigureCanvas(self.Figure)
        # add FigureCanvas to Layout
        self.FigureLayout.addWidget(self.FigureCanvas)

        # load an image
        image = Image.open(self.root + self.FileName)
        self.image = np.asarray(image)

        self.axis = self.Figure.add_subplot(1, 1, 1)
        self.axis_image = self.axis.imshow(self.image, cmap='gray')
        plt.axis('off')

        # self.resize(300, 300)

        self.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())


