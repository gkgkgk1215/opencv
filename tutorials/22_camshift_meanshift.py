import cv2
import numpy as np

col, width, row, height = -1,-1,-1,-1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
roi_hist = None

def onMouse(event, x, y, flags, param):
    global col,width,row,height, frame,frame2
    global inputmode, rectangle, roi_hist, trackWindow
    if inputmode:
        # 좌 클릭시
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x,y
            print ("left clicked")
        # 좌 클릭하는 도중 움직일 때
        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame,(col,row),(x,y),(0,255,0),2)
                cv2.imshow('frame',frame)
        # 좌 클릭 뗐을때
        elif event == cv2.EVENT_LBUTTONUP:
            inputmode = False
            rectangle = False
            cv2.rectangle(frame,(col,row),(x,y),(0,255,0),2)
            height, width = abs(row-y), abs(col-x)
            trackWindow = (col,row,width,height)
            roi = frame[row:row+height,col:col+width]
            # HSV 색공간으로 변환
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            # HSV 색공간으로 변경한 히스토그램 계산
            roi_hist = cv2.calcHist([roi],[0],None,[180],[0,180])
            # 계산된 히스토그램 노말라이즈
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return

def meanShift():
    global frame,frame2,inputmode,trackWindow,roi_hist
    try:
        cap = cv2.VideoCapture(0)
    except Exception as e:
        print(e)
        return

    ret, frame = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',onMouse,param=(frame,frame2))

    # meanShift 함수의 3번째 인자. 10회 반복 혹은 C1_o ~ C1_r의 차이가 1pt 날 때까지 작동
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if trackWindow is not None:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, trackWindow = cv2.meanShift(dst,trackWindow,termination)

            x,y,w,h = trackWindow
            # 추적된 물체 녹색 사각형으로 표시
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow('frame',frame)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        # i를 눌러서 영상을 멈춰서 roi 설정
        if k == ord('i'):
            print("Meanshift를 위한 지역을 선택하고 키를 입력해라")
            inputmode = True
            frame2 = frame.copy()

            while inputmode:
                cv2.imshow('frame',frame)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


def camShift():
    global frame,frame2,inputmode,trackWindow,roi_hist

    try:
        # 저장된 영상 불러옴
        cap = cv2.VideoCapture(0)
        cap.set(3,480)
        cap.set(4,320)
    except Exception as e:
        print(e)
        return

    ret, frame = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',onMouse,param=(frame,frame2))

    # meanShift 함수의 3번째 인자. 10회 반복 혹은 C1_o ~ C1_r의 차이가 1pt 날 때까지 작동
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if trackWindow is not None:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, trackWindow = cv2.CamShift(dst,trackWindow,termination)

            ### 이 부분 수정
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame,[pts],True,(0,255,0),2)

        cv2.imshow('frame',frame)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        # i를 눌러서 영상을 멈춰서 roi 설정
        if k == ord('i'):
            print("Select area for camshift and enter a key")
            inputmode = True
            frame2 = frame.copy()

            while inputmode:
                cv2.imshow('frame',frame)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# meanShift()
camShift()