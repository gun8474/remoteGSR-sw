import csv
import sys
import datetime
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtChart import QLineSeries, QChart
from PyQt5.QtChart import QLineSeries, QChart, QValueAxis, QDateTimeAxis
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from faceBright import facetool
from faceBright.facetool import *


QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
form_class = uic.loadUiType('gsr.ui')[0]

class CWidget(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()
        self.height = self.display_graph.height()
        self.width = self.display_graph.width()
        self.faceDetect = facetool.faceDetect(self)

    def detect(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 웹캠
            # self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            # self.cap = cv2.VideoCapture('face_bright/face_videos/IR_11-29-39.382_30fps.mp4')  # IR 비디오 읽기
            # self.cap = cv2.VideoCapture('face_videos/RGB_11-29-39.382_30fps.mp4')  # RGB 비디오 읽기
        except:
            print("error!")
        else:
            while self.cap.isOpened():
                success, orig_img = self.cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                self.orig_img, self.graph, self.mean, self.graph2, self.emotions = self.faceDetect.bright_utils(orig_img, self.height, self.width)
                self._showImage(self.orig_img, self.vid)
                self._showImage(self.graph, self.display_graph)
                self._showImage(self.graph2, self.display_graph2)
                self.save_csv()

                key = cv2.waitKey(1)
                if key & 0xFF == 27:
                    break
            self.cap.release()

    def initUI(self):
        # cam on, off button
        self.start.setCheckable(True)
        self.start.clicked.connect(self.onoffCam)
        # # self.start.clicked.connect(self.save_csv)
        self.pushButton_saveDirectory.clicked.connect(self.selectDirectory_button)
        # self.vid.setFrameShape(QFrame.Panel) # 영상 테두리 그리기
        # self.setFixedSize(1280, 750)


        self._center()
        self.setWindowTitle('OpenCV + PyQt5')
        self.show()

    def onoffCam(self, e):
        if self.start.isChecked():
            self.start.setText('stop')
            self.detect()
            #----------------------------
            # while self.cap.isOpened():
            #     success, orig_img = self.cap.read()
            # self.video.startCam()
            #-------------------------------------

        else:
            self.start.setText('start')
            self.video.stopCam()
            # self.closeEvent()

    def _showImage(self, img, display_label):
        print('showImage 1', display_label)
        if display_label is self.vid:  # 메인 라벨인 경우
            draw_img = img.copy()
            height, width, _ = img.shape
            print('check')
        elif display_label == self.display_graph:
            draw_img = img.copy()  # 그래프 그리려고 복사
            print('check 1!')
        elif display_label == self.display_graph2:
            draw_img = img.copy()  # 그래프 그리려고 복사
            print('check 2!')
        else:
            draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 3차원만 그려져서 3차원으로 변환

        qpixmap = cvtPixmap(draw_img, (display_label.width(), display_label.height())) # 이미지를 읽어서 pyqt로 보여줌
        display_label.setPixmap(qpixmap)


    def _showImage2(self, img, display_label2):
        print('showImage 2', display_label2)
        if display_label2 is self.vid:  # 메인 라벨인 경우
            draw_img = img.copy()
            height, width, _ = img.shape
            print('chek')
        elif display_label2 == self.display_graph2:
            draw_img = img.copy()  # 그래프 그리려고 복사
            print('check!')
        else:
            draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 3차원만 그려져서 3차원으로 변환

        qpixmap2 = cvtPixmap(draw_img, (display_label2.width(), display_label2.height())) # 이미지를 읽어서 pyqt로 보여줌
        display_label2.setPixmap(qpixmap2)


    def save_csv(self):
        print('arrv')
        csv_saveDir = self.label_saveDirectory.text()
        save_name = self.csvtext.toPlainText()
        self.csv_file = open(f'{csv_saveDir}/{save_name}.csv', 'a', newline='')
        csvwriter = csv.writer(self.csv_file)
        # csvwriter.writerow(['brightness'])
        csvwriter.writerow([f'{self.mean}'])
        self.csv_file.close()

    # 사용자가 선택한 csv 저장 볼더의 경로를 반환
    def selectDirectory_button(self):
        self.saved_dir = QFileDialog.getExistingDirectory(self, 'Select save directory', './')
        self.label_saveDirectory.setText(self.saved_dir)

    def _center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def recvImage(self, img):
        self.vid.setPixmap(QPixmap.fromImage(img)) # **** 영상을 frame 단위로 쪼개서 frame을 입력받아 pixmap을 사용해서 vid라는 label에 이미지를 넣는다.

    def closeEvent(self, e):
        self.video.stopCam()
        # self.deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    w.show()
    sys.exit(app.exec_())

