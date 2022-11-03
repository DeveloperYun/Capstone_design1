# 중간발표 완성버전
# 웹캠으로 이미지 딴거를
# 이미지에 이미지(웹캠) 넣기 - 성공
import cv2
import dlib
import sys
import numpy as np


# scaler = 0.2
scaler = 0.7

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("size: {0} x {1}".format(width, height))

# 영상 저장을 위한 VideoWriter 인스턴스 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test.avi', fourcc, 24, (int(width), int(height)))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        writer.write(frame)  # 프레임 저장
        cv2.imshow('Video Window', frame)

        # q 를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('test.png', frame)
            # img = cv2.imread('test.png', 1)
            break
    else:
        break

cap.release()
writer.release()  # 저장 종료

cv2.destroyAllWindows()


# cv2.imshow('test', img)
# cv2.waitKey(0)


# load overlay image
overlay = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
#overlay = cv2.imread('test.png', 1)

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

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                              mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

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

    # detect faces
    faces = detector(img)
    face = faces[0]

    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # compute center of face
    # 좌상단 우하단 , 얼굴의 중심
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)

    face_size = np.int(max(bottom_right - top_left))

    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    result = overlay_transparent(
        ori, overlay, center_x, center_y-10, overlay_size=(face_size, face_size))

    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                        pt2=(face.right(), face.bottom()),
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(
            255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.circle(img, center=tuple(top_left), radius=1, color=(
        255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1,
               color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
               color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # cv2.imshow(title, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread()의 return 값입니다.
    cv2.imshow('img', img)
    cv2.imshow('result', result)

    # 0넣으면 x클릭해야 꺼짐  1넣으면 ctrl+c해도 꺼짐
    cv2.waitKey(1)
