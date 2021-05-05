import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QApplication
from PyQt5.QtGui import QPixmap, QPainter, QPen,QImage
from train import Net
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def qt_image_to_array(img):
    img_size = img.size()
    buffer = img.constBits()
    buffer.setsize(img.byteCount())
    n_bits_buffer = len(buffer) * 8
    n_bits_image  = img_size.width() * img_size.height() * img.depth()
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), img.depth()//8),
                     buffer = buffer, 
                     dtype  = np.uint8)
    return arr[:,:,2]



class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        self.drawing = False
        self.lastPoint = QPoint()
        button_clear = QPushButton(self)
        button_clear.setText("clear")
        button_clear.move(400,400)
        button_clear.clicked.connect(self.clear)
        button_classify = QPushButton(self)
        button_classify.setText("classify")
        button_classify.move(400,450)
        button_classify.clicked.connect(self.classify)
        self.qimage = QImage(500,500, QImage.Format_RGB32)
        self.qimage.fill(Qt.black)
        self.image = QPixmap(self.qimage)
        self.setGeometry(0, 0, 500, 500)
        self.resize(self.image.width(), self.image.height())
        self.show()

        self.model = Net()
        self.model.load_state_dict(torch.load("/Users/qingyu/Documents/mnist/src/mnist_cnn.pt"))
        self.model.eval()    
    def clear(self):
        self.qimage = QImage(500,500,QImage.Format_RGB32)
        self.qimage.fill(Qt.black)
        self.image = QPixmap(self.qimage)
        self.update()

    def classify(self):
        data = qt_image_to_array(self.image.toImage())
        print(data)
        img = Image.fromarray(data)
        img = img.resize((28,28))
        img.show("123")
        tensor = transforms.Normalize((0.1307,), (0.3081,)).forward(transforms.ToTensor()(img))

        tensor = tensor.unsqueeze_(0)
        
        tensor = torch.reshape(tensor, ( 1, 1, 28,28))
        output = self.model(tensor)
        print(output.argmax(dim=1, keepdim=True) )
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 3, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainMenu = Menu()
    sys.exit(app.exec_())