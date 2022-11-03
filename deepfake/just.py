# 여기는 실험실, 이미지에 이미지(저장된 이미지) 넣기
# 이미지에 이미지 넣기 - 성공
import cv2  # 이미지 처리 라이브러리
import dlib  # 얼굴인식 라이브러리
import sys
import numpy as np  # 행렬 연산


# scaler = 0.2
scaler = 0.7

# 얼굴 디텍터 모듈 초기화
detector = dlib.get_frontal_face_detector()
# 얼굴 특징점 모듈 초기화
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load video
# cap = cv2.VideoCapture('junsu.mp4')  # 파일 이름 대신 0넣으면 웹캠이 켜진다.

# load overlay image
# 여기서 cv2.IMREAD_UNCHANGED는 이미지 파일을 alpha channel까지 포함하여 읽어들인다.
overlay = cv2.imread('111.png', cv2.IMREAD_UNCHANGED)
#overlay = cv2.imread('test.png', -1)


#
#
#
# overlay function


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if img_to_overlay_t.shape[2] == 3:
        img_to_overlay_t = cv2.cvtColor(img_to_overlay_t, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # 1 RGB인 것들은 RGBA로 변환하기
    # 2 여기서 if문으로 overlap.shape[2] 알파채널 걸러내기

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                              mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2)
               :int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img
#
#
#


while True:
    # read frame buffer from video
    # 비디오의 한 프레임씩 읽는다. 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타난다. img에 읽은 프레임이 나온다.
    # ret, img = cap.read()
    # if not ret:
    #     break

    # cv2.imread(fileName, flag) : fileName은 이미지 파일의 경로를 의미하고 flag는 이미지 파일을 읽을 때 옵션이다.
    img = cv2.imread('boy2.png', 1)
    # cv2.imshow('test', img)

    img = cv2.resize(
        img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # detect faces ,  img에서 모든 얼굴 찾기
    faces = detector(img)
    # 찾은 모든 얼굴에서 첫번째 얼굴만 가져오기
    face = faces[0]

    # predictor - img의 face 영역 안의 얼굴 특징점 찾기
    dlib_shape = predictor(img, face)
    # dlib 객체를 numpy 객체로 변환
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # compute center of face
    # 좌상단 우하단 , 얼굴의 중심
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)

    face_size = np.int(max(bottom_right - top_left))

    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    result = overlay_transparent(
        ori, overlay, center_x+10, center_y, overlay_size=(face_size, face_size))

    # visualize  얼굴 직사각형 그리기
    img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                        pt2=(face.right(), face.bottom()),
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        # 원 그리기
        cv2.circle(img, center=tuple(s), radius=1, color=(
            255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.circle(img, center=tuple(top_left), radius=1, color=(
        255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1,
               color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    # 얼굴 중심점
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
               color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # cv2.imshow(title, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread()의 return 값입니다.
    cv2.imshow('img', img)
    cv2.imshow('result', result)

    # 0넣으면 x클릭해야 꺼짐  1넣으면 ctrl+c해도 꺼짐
    cv2.waitKey(1)
