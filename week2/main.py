#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib
import sys


# In[2]:


print(cv2.__version__)       
print(np.__version__)        
print(matplotlib.__version__)


# In[3]:


import time

def calculate_fps(time_diffs):
    np_array = np.array(time_diffs)
    mean_val = np_array.mean()
    return 1/mean_val


# In[4]:


def draw_fps(frame, fps):
    fps_text = f"FPS: {fps:.1f}"
    return cv2.putText(
        frame, 
        fps_text, 
        (10, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0),  # Green text
        2, 
        cv2.LINE_AA
    )


# In[5]:


def create_mask(bgr_frame):
    # BGR to HSV
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # blue
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])

    # red
    lower_red01 = np.array([0, 120, 70])
    upper_red01 = np.array([10, 255, 255])
    lower_red02 = np.array([170,120,70])
    upper_red02 = np.array([180,255,255])

    mask01 = cv2.inRange(hsv_frame, lower_red01, upper_red01)
    mask02 = cv2.inRange(hsv_frame, lower_red02, upper_red02)

    return cv2.bitwise_or(mask01, mask02)


# In[6]:


def find_objects(mask):
    # maskClean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    threshold = 500
    objects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < threshold:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        objects.append((cx, cy, area, (x, y, w, h)))

    return objects


# In[7]:


def draw_detection(frame, detected_objects):
    for cx, cy, area, (x, y, w, h) in detected_objects:

        greenBGR = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), greenBGR, thickness)

        redBGR = (0,0,255)
        radius = 4
        cv2.circle(frame, (cx, cy), radius, redBGR, -1)

        text = f"({cx}, {cy})"
        text_position = (cx + 20, cy)

        cv2.putText(
            frame,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            redBGR,
            1,
            cv2.LINE_AA
        )

    return frame        


# In[14]:


def detect(frame):
    masked = create_mask(frame) # white objects, black background
    detected_objects = find_objects(masked) # detect boundaries
    result = draw_detection(frame, detected_objects) # draw bounding boxes on the original frame

    for cx, cy, area, (x, y, w, h) in detected_objects:
        print(f"중심: ({cx}, {cy}) | 면적: {area}px²")
    return (result, masked, detected_objects)


# In[15]:


from collections import deque


# In[16]:


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        sys.exit()

    # 너비와 높이 설정 (예: 640x480 HD 해상도)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 현재 설정된 속성값 확인해보기
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # print(f"해상도: {int(width)}x{int(height)}, FPS: {fps}")

    print("웹캠 연결 성공")

    time_diffs = deque(maxlen=30)
    start_time = time.time()

    while True:
    # ret: 프레임을 성공적으로 읽었으면 True, 아니면 False
    # frame: 읽어온 1개의 이미지 프레임 (NumPy 배열 형식)
        ret, frame = cap.read()

        # 
        if not ret:
            print("프레임을 가져올 수 없습니다. 종료합니다.")
            break


        ############
        end_time = time.time()
        time_diffs.append(end_time - start_time)
        if len(time_diffs) == 30:
            fps = calculate_fps(time_diffs)
            draw_fps(frame, fps)

        start_time = end_time
        ############

        copy_f = frame.copy()

        red_detected, mask, objects = detect(copy_f)

        maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        combined = cv2.hconcat([red_detected, maskBGR])

        # 화면에 프레임 표시
        cv2.imshow('Combined', combined)

        # 4. Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[17]:


if __name__ == "__main__":
    main()

