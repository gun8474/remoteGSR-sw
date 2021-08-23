# pip install onnxruntime
# pip install opencv-python
# pip install py-feat
# pip uninstall h5py
# conda install -c anaconda h5py


import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime
import time
import utils
import json
import face_utils
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from feat.au_detectors.StatLearning.SL_test import RandomForestClassifier, SVMClassifier, LogisticClassifier
from feat import Detector
from feat.utils import read_feat, FEAT_EMOTION_COLUMNS
from torchvision.datasets.utils import download_url
from feat.utils import get_resource_path

# Load 3DDFA(facial landmark model) parameter
bfm = face_utils._load('configs/bfm_slim.pkl')
u = bfm.get('u').astype(np.float32)  # fix bug
w_shp = bfm.get('w_shp').astype(np.float32)[..., :50]
w_exp = bfm.get('w_exp').astype(np.float32)[..., :12]
tri = bfm.get('tri')
tri = face_utils._to_ctype(tri.T).astype(np.int32)
keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
w = np.concatenate((w_shp, w_exp), axis=1)
w_norm = np.linalg.norm(w, axis=0)
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]

# params normalization config
r = face_utils._load('configs/param_mean_std_62d_120x120.pkl')
param_mean = r.get('mean')
param_std = r.get('std')

# cv2.imshow에 사용할 색상 정의
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)

if __name__ == "__main__":

    # Check if model files have been downloaded. Otherwise download model.
    # get model url.
    # 경원오빠 코드 - AU만 계산
    au_model = 'logistic'
    au_regressor = LogisticClassifier()

    # video에서 face, landmark, au, emotion 검출
    face_model = "retinaface"
    landmark_model = "mobilenet"
    # au_model = "rf"
    emotion_model = "resmasknet"
    # detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model,
    #                     emotion_model=emotion_model)

    # video에서 emotion만 검출
    detector = Detector(face_model=face_model, landmark_model=landmark_model, emotion_model=emotion_model)

    # load onnx version of BFM
    session = onnxruntime.InferenceSession('onnx/facedetector.onnx', None)
    lm_session = onnxruntime.InferenceSession('onnx/TDDFA.onnx', None)
    input_name = session.get_inputs()[0].name
    class_names = ["BACKGROUND", "FACE"]

    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 노트북 카메라만 있는 경우
    # cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # 노트북 카메라
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # RGB 카메라
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # IR 카메라

    cap = cv2.VideoCapture('face_videos/IR_11-29-39.382_30fps.mp4')  # IR 비디오 읽기
    # cap = cv2.VideoCapture('face_videos/RGB_11-29-39.382_30fps.mp4')  # RGB 비디오 읽기
    preTime = 0

    while cap.isOpened():
        success, orig_image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # try:

        # 논문 저자가 모델을 학습할 때 얼굴 이미지를 -1~1 사이의 값으로 정규화함
        # -1~1 사이값을 가질 때 모델 학습이 더 잘 되기 때문
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # CV BGR -> RGB(모델에 맞게)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        confidences, boxes = session.run(None, {input_name: image})
        boxes, labels, probs = utils.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, 0.9)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            box_margin = int((box[2] - box[0]) / 6)
            # add box margin for accurate face landmark inference
            box = [box[0] - box_margin, box[1] - box_margin, box[2] + box_margin, box[3] + box_margin]
            cropped_img = orig_image[box[1]:box[3], box[0]:box[2]].copy()
            cropped_img = cv2.resize(cropped_img, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
            cropped_img = cropped_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            cropped_img = (cropped_img - 127.5) / 128.0
            param = lm_session.run(None, {'input': cropped_img})[0]
            param = param.flatten().astype(np.float32)
            param = param * param_std + param_mean  # re-scale
            # vers = face_utils.recon_vers2D([param], [[box[0],box[1],box[2],box[3]]], u_base, w_shp_base, w_exp_base)  # 건영오빠가 새로 주신 코드
            # vers = face_utils.recon_vers([param], [[box[0], box[1], box[2], box[3]]], u_base, w_shp_base, w_exp_base)[0]  # 기존 코드
            vers = \
            face_utils.recon_vers2D([param], [[box[0], box[1], box[2], box[3]]], u_base, w_shp_base, w_exp_base)[
                0]  # 내 코드
            vers = np.array(vers)
            # print("type: ", type(vers))
            # print("shape: ", np.shape(vers))

            # ===========================피부 영역 정의 후 밝기값 추출=========================================
            img = orig_image.copy()  # 이미지 복사, BGR 이미지
            mask_img = np.zeros((480, 640))  # 피부영역 정의용 mask


            # 왼쪽 뺨 - 0, 31, 48, 6, 5
            Lcheek = np.array([[vers[0, 0], vers[0, 1]], [vers[31, 0], vers[31, 1]], [vers[48, 0], vers[48, 1]],
                               [vers[6, 0], vers[6, 1]], [vers[5, 0], vers[5, 1]]], np.int32)
            # 오른쪽 뺨 - 16, 35, 54, 11, 12
            Rcheek = np.array([[vers[16, 0], vers[16, 1]], [vers[35, 0], vers[35, 1]], [vers[54, 0], vers[54, 1]],
                               [vers[11, 0], vers[11, 1]], [vers[12, 0], vers[12, 1]]], np.int32)
            # 코 - 27, 31, 33, 35
            nose = np.array([[vers[27, 0], vers[27, 1]], [vers[31, 0], vers[31, 1]],
                             [vers[33, 0], vers[33, 1]], [vers[35, 0], vers[35, 1]]], np.int32)
            # 턱 - 48, 59, 55, 54, 11, 8, 6
            jaw = np.array([[vers[48, 0], vers[48, 1]], [vers[59, 0], vers[59, 1]], [vers[55, 0], vers[55, 1]],
                            [vers[54, 0], vers[54, 1]], [vers[11, 0], vers[11, 1]], [vers[8, 0], vers[8, 1]],
                            [vers[6, 0], vers[6, 1]]], np.int32)
            # 이마 - 18, 25
            fore = np.array([[vers[18, 0], vers[18, 1]], [vers[25, 0], vers[25, 1]],
                             [vers[25, 0], vers[25, 1] + (vers[59, 1] - vers[6, 1])],
                             [vers[18, 0], vers[18, 1] + (vers[55, 1] - vers[10, 1])]], np.int32)

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

            # 정의한 피부영역 시각화
            cv2.polylines(orig_image, [Lcheek], True, green_color)
            cv2.polylines(orig_image, [Rcheek], True, green_color)
            cv2.polylines(orig_image, [nose], True, green_color)
            cv2.polylines(orig_image, [jaw], True, green_color)
            cv2.polylines(orig_image, [fore], True, green_color)
            # orig_image = orig_image[box[1]:box[3], box[0]:box[2]]  # 얼굴 부분만 crop, 얼굴 크기가 프레임별로 변해서 정신없음

            # ======================================= Detect Action Unit =======================================
            vers = np.array(vers)
            aligned_img, new_lands = face_utils.align_face_68pts(orig_image, vers.flatten(), 2.5, img_size=112)
            hull = ConvexHull(new_lands)
            mask = grid_points_in_poly(shape=np.array(aligned_img).shape,
                                       # for some reason verts need to be flipped
                                       verts=list(
                                           zip(new_lands[hull.vertices][:, 1], new_lands[hull.vertices][:, 0]))
                                       )

            mask[0:np.min([new_lands[0][1], new_lands[16][1]]), new_lands[0][0]:new_lands[16][0]] = True
            aligned_img[~mask] = 0
            resized_face_np = aligned_img
            convex_hull = cv2.cvtColor(resized_face_np, cv2.COLOR_BGR2RGB)
            hogs = face_utils.extract_hog(frame=convex_hull)

            # AU01(SUR[0]), AU02(SUR[1]), AU04, AU05(SUR[3]), AU06,
            # AU07, AU09, AU10, AU11, AU12,
            # AU14, AU15, AU17, AU20, AU23,
            # AU24, AU25, AU26(SUR[17]), AU28, AU43
            # AU01-Inner Brow Raiser, AU02-Outer Brow Raiser, AU05-Upper Lid Raiser, AU26-Jaw Drop
            au_occur = au_regressor.detect_au(frame=hogs, landmarks=new_lands)[0]
            print(f"{au_occur[0]:.2f}, {au_occur[1]:.2f}, {au_occur[3] * 10:.2f}, {au_occur[17]:.2f}")

            # ======================================= Detect Action Unit =======================================



            # ====================================== Detect facial experssions ===================================
            # process_frame_emotion() 함수 너무 느림
            pred = detector.process_frame_emotion(orig_image)  # 입력값인 orig_image는 np.array
            print(pred)


            # Draw face landmark(by kunyoung)
            # vers = vers[0]
            for i in range(68):
                cv2.circle(orig_image, (int(vers[i, 0]), int(vers[i, 1])), 1, (0, 255, 0), -1,
                           cv2.LINE_AA)  # 랜드마크 그리기

            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)  # 얼굴 bbox 그리기
            # cv2.putText(orig_image, label, (box[0], box[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)  # Face 확률 그리기(띄우기)


        # except:
        #     print("Face detecting error!")

        curTime = time.time()
        sec = curTime - preTime
        preTime = curTime
        fps = 1 / (sec)

        cv2.putText(orig_image, "FPS : %0.1f" % fps, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))  # FPS 출력
        cv2.imshow('face detection', orig_image)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()