# pip install onnxruntime
# pip install opencv-python

import cv2
import numpy as np
import onnxruntime


# from graphDraw import *
from faceBright import face_utils
from faceBright import utils
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Thread
from PyQt5.QtCore import QThread, pyqtSignal

# 이미지를 읽어서 pyqt로 보여주는 함수
def cvtPixmap(frame, img_size):
    frame = cv2.resize(frame, img_size)
    height, width, channel = frame.shape
    bytesPerLine = 3 * width
    qImg = QImage(frame.data,
                  width,
                  height,
                  bytesPerLine,
                  QImage.Format_RGB888).rgbSwapped()
    qpixmap = QPixmap.fromImage(qImg)

    return qpixmap

class faceDetect(QObject):
    # pyqtSignal()을 이용해서 사용자 정의 시그널을 만들고, 특정 이벤트가 발생했을 때 이 시그널이 방출되도록 함
    sendImage = pyqtSignal(QImage)
    #-------------------------------
    # sendData = pyqtSignal(float)
    #-----------------------------------
    def __init__(self, Widget):
        super().__init__()
        # self.widget = gsr_sw.CWidget()
        self.Widget = Widget
        self.sendImage.connect(self.Widget.recvImage)
        #----------------------------------------------------
        # self.sendData.connect(self.Widget.appendData)

        # self.graph = graphDraw(self)
        #-----------------------------------------------------
        # Load 3DDFA(facial landmark model) parameter
        self.bfm = face_utils._load('faceBright/configs/bfm_slim.pkl')
        # self.bfm = face_utils._load('/configs/bfm_slim.pkl')
        self.u = self.bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = self.bfm.get('w_shp').astype(np.float32)[..., :50]
        self.w_exp = self.bfm.get('w_exp').astype(np.float32)[..., :12]
        self.tri = self.bfm.get('tri')
        self.tri = face_utils._to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = self.bfm.get('keypoints').astype(np.long)  # fix bug
        self.w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(self.w, axis=0)
        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

        # params normalization config
        self.r = face_utils._load('faceBright/configs/param_mean_std_62d_120x120.pkl')
        # self.r = face_utils._load('/configs/param_mean_std_62d_120x120.pkl')
        self.param_mean = self.r.get('mean')
        self.param_std = self.r.get('std')
        self.emotion_dict = {0: 'NEUTRAL', 1: 'HAPPY', 2: 'SURPRISE', 3: 'SADNESS', 4: 'ANGER', 5: 'DISGUST', 6: 'FEAR'}

        self.red_color = (0, 0, 255)
        self.green_color = (0, 255, 0)
        self.blue_color = (255, 0, 0)

        # load onnx version of BFM
        self.session = onnxruntime.InferenceSession('faceBright/onnx/facedetector.onnx', None)
        self.lm_session = onnxruntime.InferenceSession('faceBright/onnx/TDDFA.onnx', None)
        self.fer_session = onnxruntime.InferenceSession('faceBright/onnx/FER.onnx', None)   # emotion 검출용
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = ["BACKGROUND", "FACE"]

        self.gsr_data = []
        self.emotions = []


    def bright_utils(self, orig_img, height, width):
        print('얼굴 밝기값 검출 시작')
        try:
            self.image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # CV BGR -> RGB(모델에 맞게)
            self.image = cv2.resize(self.image, (320, 240))
            self.image_mean = np.array([127, 127, 127])
            self.image = (self.image - self.image_mean) / 128
            self.image = np.transpose(self.image, [2, 0, 1])
            self.image = np.expand_dims(self.image, axis=0)
            self.image = self.image.astype(np.float32)


            self.confidences, self.boxes = self.session.run(None, {self.input_name: self.image})
            self.boxes, self.labels, self.probs = utils.predict(orig_img.shape[1], orig_img.shape[0], self.confidences, self.boxes, 0.9)
            print('box shape',self.boxes.shape[0] )

            try:
                # for i in range(self.boxes.shape[0]):
                i = 0
                box = self.boxes[i, :]
                label = f"{self.class_names[self.labels[i]]}: {self.probs[i]:.2f}"
                box_margin = int((box[2] - box[0]) / 6)
                # add box margin for accurate face landmark inference
                box = [box[0] - box_margin, box[1] - box_margin, box[2] + box_margin, box[3] + box_margin]
                cropped_img = orig_img[box[1]:box[3], box[0]:box[2]].copy()
                cropped_img = cv2.resize(cropped_img, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
                cropped_img = cropped_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
                cropped_img = (cropped_img - 127.5) / 128.0
                param = self.lm_session.run(None, {'input': cropped_img})[0]
                param = param.flatten().astype(np.float32)
                param = param * self.param_std + self.param_mean  # re-scale
                vers = face_utils.recon_vers([param], [[box[0], box[1], box[2], box[3]]], self.u_base, self.w_shp_base, self.w_exp_base)[0]


                # 이미지 복사
                img = orig_img.copy()  # BGR 이미지
                mask_img = np.zeros((480, 640))


                # 왼쪽 뺨 - 0, 31, 48, 6, 5
                Lcheek = np.array([[vers[0, 0], vers[1, 0]], [vers[0, 31], vers[1, 31]],
                                   [vers[0, 48], vers[1, 48]], [vers[0, 6], vers[1, 6]], [vers[0, 5], vers[1, 5]]],
                                  np.int32)
                # 오른쪽 뺨 - 16, 35, 54, 11, 12
                Rcheek = np.array([[vers[0, 16], vers[1, 16]], [vers[0, 35], vers[1, 35]],
                                   [vers[0, 54], vers[1, 54]], [vers[0, 11], vers[1, 11]], [vers[0, 12], vers[1, 12]]],
                                  np.int32)
                # 코 - 27, 31, 33, 35
                nose = np.array([[vers[0, 27], vers[1, 27]], [vers[0, 31], vers[1, 31]],
                                 [vers[0, 33], vers[1, 33]], [vers[0, 35], vers[1, 35]]], np.int32)
                # 턱 - 48, 59, 55, 54, 11, 6
                jaw = np.array([[vers[0, 48], vers[1, 48]], [vers[0, 59], vers[1, 59]], [vers[0, 55], vers[1, 55]],
                                [vers[0, 54], vers[1, 54]], [vers[0, 11], vers[1, 11]], [vers[0, 6], vers[1, 6]]], np.int32)
                # 이마 - 18, 25
                fore = np.array([[vers[0, 18], vers[1, 18]], [vers[0, 25], vers[1, 25]],
                                 [vers[0, 25], vers[1, 25] + (vers[1, 59] - vers[1, 6])],
                                 [vers[0, 18], vers[1, 18] + (vers[1, 55] - vers[1, 10])]], np.int32)

                # 피부 영역 흰색으로 채우기
                cv2.fillConvexPoly(mask_img, Lcheek, 1)
                cv2.fillConvexPoly(mask_img, Rcheek, 1)
                cv2.fillConvexPoly(mask_img, nose, 1)
                cv2.fillConvexPoly(mask_img, jaw, 1)
                cv2.fillConvexPoly(mask_img, fore, 1)

                mask_img = np.array(mask_img, dtype=np.int8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                skin = cv2.bitwise_and(gray, gray, mask=mask_img)  # 정의한 피부 영역, (480, 640, 3)

                # 평균 밝기값 구하기
                skin_num = mask_img.sum()  # 피부 픽셀 개수
                skin_sum = skin.sum()
                mean = skin_sum / skin_num  # 얼굴 평균 밝기값
                print("얼굴 평균 밝기값: {:.4f}".format(mean))


                # emotion 검출
                temp = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(temp, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
                img = img.astype(np.float32)[np.newaxis, np.newaxis, ...]  # RGB카메라의 경우 np.array(1, 1, 48, 48)
                img = (img - 127.5) / 128.0
                inp_dct = {'input': img}
                log_ps = self.fer_session.run(None, inp_dct)[0]
                log_ps = np.exp(log_ps[0])  # np.array 타입

                np.set_printoptions(precision=2, suppress=True)
                emo_4 = np.append(log_ps[:3], log_ps[4])
                print("4 EMOTIONS: ", emo_4)

                emo_max = np.argmax(emo_4)  # 7가지 중 최댓값
                emo_outputs = emo_4 * 100
                emo_outputs = np.array(emo_outputs, dtype=np.int32)
                print("OUTPUTS: ", emo_outputs)

                # 시각화
                cv2.polylines(orig_img, [Lcheek], True, self.green_color)
                cv2.polylines(orig_img, [Rcheek], True, self.green_color)
                cv2.polylines(orig_img, [nose], True, self.green_color)
                cv2.polylines(orig_img, [jaw], True, self.green_color)
                cv2.polylines(orig_img, [fore], True, self.green_color)
                cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                # # 68개 랜드마크 그리기
                # for i in range(68):
                #     cv2.circle(orig_img, (int(vers[0, i]), int(vers[1, i])), 1, (0, 255, 0), -1, cv2.LINE_AA)

                # 그래프 그리기 위한 도화지 설정
                plot_canvas = np.zeros((height, width, 3), np.uint8)
                plot_canvas2 = np.zeros((height, width, 3), np.uint8)

                # 프레임별로 계산된 값 append
                self.gsr_data.append(mean)
                self.emotions.append(emo_outputs)

                # 정해준 범위 이상이면 데이터 1개씩 삭제
                while len(self.gsr_data) > width/2:
                    del self.gsr_data[0]
                    del self.emotions[0]

                # 평균 밝기값, emotion 그래프 그리기
                for i in range(len(self.gsr_data) - 1):
                    print("gsr: ", self.gsr_data[i])
                    print("emotions: ", self.emotions[i])
                    cv2.line(plot_canvas, (i, height - int(self.gsr_data[i])), (i + 1, height - int(self.gsr_data[i + 1])), (0, 255, 0), 2)
                    cv2.line(plot_canvas2, (i, height - int(self.emotions[i][0])),
                             (i + 1, height - int(self.emotions[i + 1][0])), (255, 255, 255), 1)  # neutral
                    cv2.line(plot_canvas2, (i, height - int(self.emotions[i][1])), (i + 1, height - int(self.emotions[i + 1][1])),
                             (0, 0, 255), 2)  # happy
                    cv2.line(plot_canvas, (i, height - int(self.emotions[i][2])), (i + 1, height - int(self.emotions[i + 1][2])),
                             (0, 35, 35), 2)  # surprise
                    cv2.line(plot_canvas, (i, height - int(self.emotions[i][3])), (i + 1, height - int(self.emotions[i + 1][3])),
                             (35, 0, 35), 2)  # anger


                cv2.putText(orig_img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)  # line color and type
            except:
                print("box error!")
        except:
            print("Face detecting error!")

        return orig_img, plot_canvas, mean, plot_canvas2, emo_outputs

