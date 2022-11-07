from django.shortcuts import render
from django.http import HttpResponse

import cv2
import dlib
import sys
import numpy as np

# overlay function, 이미지(웹캠)을 이미지에 띄우는 것은 구글링을 통해 찾았다함.
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

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2)
               :int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

def main(request):
    # HttpResponse는 요청에 대한 응답을 할때 사용한다

    # 이미지 크기를 조정해준다 0.7은 10분의 7로 줄여주기 위한 변수. cv2.resize에서 쓰인다.
    scaler = 5.0
    # 얼굴 디텍터 모듈 초기화
    detector = dlib.get_frontal_face_detector()
    # 얼굴 특징점 모듈 초기화 shape_predictor_68_face_landmarks.dat 이 파일이 있어야 실행가능
    # shape_predictor_68_face_landmarks.dat 는 머신러닝으로 학습된 모델
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 웹캠이 켜진다.
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
    overlay = cv2.imread('test.png', cv2.IMREAD_UNCHANGED) # 캠으로 찍은 내 사진
    
    while True:
        # cv2.imread(fileName, flag) : fileName은 이미지 파일의 경로를 의미하고 flag는 이미지 파일을 읽을 때 옵션이다.
        img = cv2.imread('boy.png', 1)

        # img에 (int(img.shape[1] * scaler), int(img.shape[0] * scaler)) 크기로 조절
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        # detect faces, img에서 모든 얼굴 찾기
        faces = detector(img)
        # 찾은 모든 얼굴에서 첫번째 얼굴만 가져오기
        face = faces[0]

        # img의 face 영역 안의 얼굴 특징점 찾기
        dlib_shape = predictor(img, face)
        # dlib 객체를 numpy 객체로 변환(연산을 쉽게 하기 위해)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # compute center of face
        # 좌상단 우하단 , 얼굴의 중심
        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = np.int(max(bottom_right - top_left))

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # 복사한 이미지를 센터x, 센터y 중심으로 넣고 overlay_size 만큼 resize해서
        # 원본 이미지에 넣어준다. 크기는 얼굴 크기만큼 resize해주는 것이다.
        result = overlay_transparent(
            ori, overlay, center_x, center_y-10, overlay_size=(face_size*2, face_size*2))

        # visualize , 직사각형 그리기
        img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                            pt2=(face.right(), face.bottom()),
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 특징점 그리기, 68개의 점을 찾아줌
        for s in shape_2d:
            # 얼굴에 원그리기
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 왼쪽 위 파란점
        cv2.circle(img, center=tuple(top_left), radius=1, color=(
            255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 얼굴 오른쪽 아래 파란점
        cv2.circle(img, center=tuple(bottom_right), radius=1,
                color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 모든 특징점의 평균을 구해서 얼굴의 중심을 구하기
        cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # cv2.imshow(title, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread()의 return 값입니다.
        # 모니터에 이미지를 보여주는 함수
        cv2.imshow('img', img)
        cv2.imshow('result', result)

        # 0넣으면 x클릭해야 꺼짐  1넣으면 ctrl+c해도 꺼짐
        cv2.waitKey(1)  # 1밀리세컨드만큼 대기. 이걸 넣어야 동영상이 제대로 보임
        
        # 사용자가 자기 사진 찍는건 Media 파일
        # 해당 경로에 결과물 저장
        cv2.imwrite("static/img/result.png",result)
        break

    # 배경제거 알고리즘 적용.
    # imageUrl = 'test.png'
    # imgres = cv2.imread(imageUrl)

    # mask = np.zeros(imgres.shape[:2],np.uint8)

    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)

    # rect = (1,1,665,344)
    # cv2.grabCut(imgres,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # imgres = imgres*mask2[:,:,np.newaxis]

    # tmp = cv2.cvtColor(imgres, cv2.COLOR_BGR2GRAY)
    # _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    # b, g, r = cv2.split(imgres)
    # rgba = [b,g,r, alpha]
    # dst = cv2.merge(rgba,4)

    # cv2.imwrite("static/img/result.png", dst)

    return render(request, 'mainfunc/main.html')

def main2(request):
    # HttpResponse는 요청에 대한 응답을 할때 사용한다

    # 이미지 크기를 조정해준다 0.7은 10분의 7로 줄여주기 위한 변수. cv2.resize에서 쓰인다.
    scaler = 5.0
    # 얼굴 디텍터 모듈 초기화
    detector = dlib.get_frontal_face_detector()
    # 얼굴 특징점 모듈 초기화 shape_predictor_68_face_landmarks.dat 이 파일이 있어야 실행가능
    # shape_predictor_68_face_landmarks.dat 는 머신러닝으로 학습된 모델
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    '''
    # 웹캠이 켜진다.
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
    '''
    overlay = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
    
    while True:
        # cv2.imread(fileName, flag) : fileName은 이미지 파일의 경로를 의미하고 flag는 이미지 파일을 읽을 때 옵션이다.
        img = cv2.imread('boy2.png', 1)

        # img에 (int(img.shape[1] * scaler), int(img.shape[0] * scaler)) 크기로 조절
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        # detect faces, img에서 모든 얼굴 찾기
        faces = detector(img)
        # 찾은 모든 얼굴에서 첫번째 얼굴만 가져오기
        face = faces[0]

        # img의 face 영역 안의 얼굴 특징점 찾기
        dlib_shape = predictor(img, face)
        # dlib 객체를 numpy 객체로 변환(연산을 쉽게 하기 위해)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # compute center of face
        # 좌상단 우하단 , 얼굴의 중심
        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = np.int(max(bottom_right - top_left))

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # 복사한 이미지를 센터x, 센터y 중심으로 넣고 overlay_size 만큼 resize해서
        # 원본 이미지에 넣어준다. 크기는 얼굴 크기만큼 resize해주는 것이다.
        result = overlay_transparent(
            ori, overlay, center_x, center_y-10, overlay_size=(face_size, face_size))

        # visualize , 직사각형 그리기
        img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                            pt2=(face.right(), face.bottom()),
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 특징점 그리기, 68개의 점을 찾아줌
        for s in shape_2d:
            # 얼굴에 원그리기
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 왼쪽 위 파란점
        cv2.circle(img, center=tuple(top_left), radius=1, color=(
            255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 얼굴 오른쪽 아래 파란점
        cv2.circle(img, center=tuple(bottom_right), radius=1,
                color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 모든 특징점의 평균을 구해서 얼굴의 중심을 구하기
        cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # cv2.imshow(title, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread()의 return 값입니다.
        # 모니터에 이미지를 보여주는 함수
        cv2.imshow('img', img)
        cv2.imshow('result', result)

        # 0넣으면 x클릭해야 꺼짐  1넣으면 ctrl+c해도 꺼짐
        cv2.waitKey(1)  # 1밀리세컨드만큼 대기. 이걸 넣어야 동영상이 제대로 보임
        
        # 사용자가 자기 사진 찍는건 Media 파일
        cv2.imwrite("static/img/result.png",result)
        break
    return render(request, 'mainfunc/main.html')

def main3(request):

    # HttpResponse는 요청에 대한 응답을 할때 사용한다

    # 이미지 크기를 조정해준다 0.7은 10분의 7로 줄여주기 위한 변수. cv2.resize에서 쓰인다.
    scaler = 5.0
    # 얼굴 디텍터 모듈 초기화
    detector = dlib.get_frontal_face_detector()
    # 얼굴 특징점 모듈 초기화 shape_predictor_68_face_landmarks.dat 이 파일이 있어야 실행가능
    # shape_predictor_68_face_landmarks.dat 는 머신러닝으로 학습된 모델
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    '''
    # 웹캠이 켜진다.
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
    '''
    overlay = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
    
    while True:
        # cv2.imread(fileName, flag) : fileName은 이미지 파일의 경로를 의미하고 flag는 이미지 파일을 읽을 때 옵션이다.
        img = cv2.imread('boy3.png', 1)

        # img에 (int(img.shape[1] * scaler), int(img.shape[0] * scaler)) 크기로 조절
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        # detect faces, img에서 모든 얼굴 찾기
        faces = detector(img)
        # 찾은 모든 얼굴에서 첫번째 얼굴만 가져오기
        face = faces[0]

        # img의 face 영역 안의 얼굴 특징점 찾기
        dlib_shape = predictor(img, face)
        # dlib 객체를 numpy 객체로 변환(연산을 쉽게 하기 위해)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # compute center of face
        # 좌상단 우하단 , 얼굴의 중심
        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = np.int(max(bottom_right - top_left))

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # 복사한 이미지를 센터x, 센터y 중심으로 넣고 overlay_size 만큼 resize해서
        # 원본 이미지에 넣어준다. 크기는 얼굴 크기만큼 resize해주는 것이다.
        result = overlay_transparent(
            ori, overlay, center_x, center_y-10, overlay_size=(face_size, face_size))

        # visualize , 직사각형 그리기
        img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                            pt2=(face.right(), face.bottom()),
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 특징점 그리기, 68개의 점을 찾아줌
        for s in shape_2d:
            # 얼굴에 원그리기
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 왼쪽 위 파란점
        cv2.circle(img, center=tuple(top_left), radius=1, color=(
            255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 얼굴 오른쪽 아래 파란점
        cv2.circle(img, center=tuple(bottom_right), radius=1,
                color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 모든 특징점의 평균을 구해서 얼굴의 중심을 구하기
        cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # cv2.imshow(title, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread()의 return 값입니다.
        # 모니터에 이미지를 보여주는 함수
        cv2.imshow('img', img)
        cv2.imshow('result', result)

        # 0넣으면 x클릭해야 꺼짐  1넣으면 ctrl+c해도 꺼짐
        cv2.waitKey(1)  # 1밀리세컨드만큼 대기. 이걸 넣어야 동영상이 제대로 보임
        
        # 사용자가 자기 사진 찍는건 Media 파일
        cv2.imwrite("static/img/result.png",result)
        break
    return render(request, 'mainfunc/main.html')

def main4(request):
    # HttpResponse는 요청에 대한 응답을 할때 사용한다

    # 이미지 크기를 조정해준다 0.7은 10분의 7로 줄여주기 위한 변수. cv2.resize에서 쓰인다.
    scaler = 5.0
    # 얼굴 디텍터 모듈 초기화
    detector = dlib.get_frontal_face_detector()
    # 얼굴 특징점 모듈 초기화 shape_predictor_68_face_landmarks.dat 이 파일이 있어야 실행가능
    # shape_predictor_68_face_landmarks.dat 는 머신러닝으로 학습된 모델
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    '''
    # 웹캠이 켜진다.
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
    '''
    overlay = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
    
    while True:
        # cv2.imread(fileName, flag) : fileName은 이미지 파일의 경로를 의미하고 flag는 이미지 파일을 읽을 때 옵션이다.
        img = cv2.imread('boy4.png', 1)

        # img에 (int(img.shape[1] * scaler), int(img.shape[0] * scaler)) 크기로 조절
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        # detect faces, img에서 모든 얼굴 찾기
        faces = detector(img)
        # 찾은 모든 얼굴에서 첫번째 얼굴만 가져오기
        face = faces[0]

        # img의 face 영역 안의 얼굴 특징점 찾기
        dlib_shape = predictor(img, face)
        # dlib 객체를 numpy 객체로 변환(연산을 쉽게 하기 위해)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        # compute center of face
        # 좌상단 우하단 , 얼굴의 중심
        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)

        face_size = np.int(max(bottom_right - top_left))

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # 복사한 이미지를 센터x, 센터y 중심으로 넣고 overlay_size 만큼 resize해서
        # 원본 이미지에 넣어준다. 크기는 얼굴 크기만큼 resize해주는 것이다.
        result = overlay_transparent(
            ori, overlay, center_x, center_y-10, overlay_size=(face_size, face_size))

        # visualize , 직사각형 그리기
        img = cv2.rectangle(img, pt1=(face.left(), face.top()),
                            pt2=(face.right(), face.bottom()),
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 특징점 그리기, 68개의 점을 찾아줌
        for s in shape_2d:
            # 얼굴에 원그리기
            cv2.circle(img, center=tuple(s), radius=1, color=(
                255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # 얼굴 왼쪽 위 파란점
        cv2.circle(img, center=tuple(top_left), radius=1, color=(
            255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 얼굴 오른쪽 아래 파란점
        cv2.circle(img, center=tuple(bottom_right), radius=1,
                color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # 모든 특징점의 평균을 구해서 얼굴의 중심을 구하기
        cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # cv2.imshow(title, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread()의 return 값입니다.
        # 모니터에 이미지를 보여주는 함수
        cv2.imshow('img', img)
        cv2.imshow('result', result)

        # 0넣으면 x클릭해야 꺼짐  1넣으면 ctrl+c해도 꺼짐
        cv2.waitKey(1)  # 1밀리세컨드만큼 대기. 이걸 넣어야 동영상이 제대로 보임
        
        # 사용자가 자기 사진 찍는건 Media 파일
        cv2.imwrite("static/img/result.png",result)
        break
    return render(request, 'mainfunc/main.html')
