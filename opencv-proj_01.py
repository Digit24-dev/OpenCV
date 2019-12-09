# 프로그래밍 언어 과제 - 파이썬으로 opencv를 활용한 손가락 갯수 세는 프로그램.

from __future__ import print_function
import numpy as np
import cv2

lowerBound = np.array([0, 133, 77])
upperBound = np.array([255, 173, 127])
# 각 numpy 변수들은 c++언어의 Matrix 타입과 같은 역할 - cap.set의 변수와 같은 640x480 의 크기와 0~255의 RGB 색상 값을 가짐.


def showcam():
    try:
        print('open cam')
        cap = cv2.VideoCapture(0)
    except:
        print('Not working')
        return
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()              # 캠으로 부터 화면 가져와서 frame에 저장
        frame = cv2.flip(frame, 1)           # frame 화면 좌우 반전
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)            # 사람의 피부 색상 톤 을 가져오기 위해 ycrcb톤 사용

        # 위에서 정의한 upper/lowerBound 의 범위 값 내의 이미지만 가져옴
        mask_hand = cv2.inRange(ycrcb, lowerBound, upperBound)
        # ret1, thr = cv2.threshold(mask_hand, 127, 255, 0)      # -> 임계값 정해 이미지 생성하는 함수

        # 경계면을 그리기 위한 contour 함수 및 저장할 변수 선언
        contoured_img = frame
        contours, _ = cv2.findContours(mask_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 이진화 이미지에서 경계면으로부터의 거리를 측정하기 위한 함수를 불러와 dist_transform에 저장. 출력 변수 값은 float타입
        # 출력 변수가 float 타입이기 때문에 normalize 함수를 사용해 255~0 사이의 값을 갖게 한다.
        dist_transform = cv2.distanceTransform(mask_hand, cv2.DIST_L2, 5)
        result = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # 받아온 result 좌표 값 들 중 가장 큰 거리 값과 그 값의 좌표 값을 가져옴 / 최솟값은 사용하지 않음
        _, max_dst, _, max_dst_coordinates = cv2.minMaxLoc(result, mask_hand)

        # mask_hand 프레임에 가장 먼 좌표 값을 중심으로 max_dst 만큼의 원을 그림
        mask_hand = cv2.circle(mask_hand, max_dst_coordinates, int(max_dst), (100, 100, 100), 2)

        # contours 알고리즘
        if len(contours) > 0:
            for i in range(len(contours)):
                # Get area value
                area = cv2.contourArea(contours[i])
                if area > 100:  # minimum yellow area
                    rect = cv2.minAreaRect(contours[i])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    contoured_img = cv2.drawContours(contoured_img, [box], -1, (0, 255, 0), 3)

        # convex_hull 알고리즘
        # list형 변수 hull_list 선언
        hull_list = []
#        for i in range(len(contours)):
#            hull = cv2.convexHull(contours[i])
#            hull_list.append(hull)

        # drawing 프레임에 contours 알고리즘으로 찾아낸 경계선 그림
        drawing = np.zeros((mask_hand.shape[0], mask_hand.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (0, 0, 255)
            cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(drawing, hull_list, i, color)

        # 손가락 갯수 세기
        count = getFingerCount(mask_hand, max_dst_coordinates, int(max_dst))
        cv2.putText(mask_hand, "finger count : "+str(count), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 2)

        # ret - 캠 화면을 가져오지 못했을 경우 => 에러 및 종료
        if not ret:
            print('error')
            break

        # 윈도우 창 출력
        cv2.imshow('bitwise', mask_hand)
        cv2.imshow('cam_load', frame)
        # cv2.imshow('convex_hull', drawing)
        cv2.imshow('dist', result)

        # 키 입력 값 = q일 경우 종료
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    # cap 해제, 모든 창 종료
    cap.release()
    cv2.destroyAllWindows()


def getFingerCount(mask, center, radius, scale=2.0):
    cimage = mask
    cv2.circle(cimage, center, int(radius*scale), (255, 255, 255))
    contours1, hierachy = cv2.findContours(cimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fingercount = 0
    b = np.array(contours1)

    if b.size == 0:
        return -1

    # 좌표 값 추출하여 0에서 1로 바뀌는 알고리즘 필요

    return fingercount - 1


getFingerCount.flag = 0

# 실행
showcam()

# Coding Log
#    11.06  convex_hull 까지 적용
#    11.11 morphology 알고리즘은 최적화를 위한 알고리즘 굳이 사용 안해도 될듯
#    11.17 평균 값으로 손의 중앙점을 찾아보려 했으나 잡음이 너무 심하다.
#    11.20 distanceTransform 함수로 이진화 이미지의 픽셀값의 거리(값이 1인 픽셀의 배경화면 픽셀 값 0 으로부터의 거리) 측정
#    normalize로 평균화 후, minMaxLoc 함수로 distanceTranform 결과 값의 최댓값을 구한 후 가장 거리가 먼 곳을 중심점으로 지정
#    지정한 곳을 중심으로 한 원을 그림

# TODO : 거리 알고리즘을 통한 원으로 겹치는 구간에 따른 손가락 갯수 측정 알고리즘 구현
#        *추가로 haar_cascade 알고리즘으로 얼굴영역 지우기 필요*
#        히스토그램 방식이 조명에 따라 바뀌는 색감에 덜 민감해서 사용하기 좋을 듯..(배경이미지 대조 방식)
#        12.07 numpy 배열 [[[[1, 1]]]에 접근하는 방법 찾기.
