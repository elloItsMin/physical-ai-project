#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib
import sys
print(cv2.__version__)       
print(np.__version__)        
print(matplotlib.__version__)


# In[2]:


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# In[3]:


def apply_blur(gray_frame):
    return cv2.GaussianBlur(gray_frame, (5,5), 0)


# In[4]:


def detect_edges(blur_frame):
    return cv2.Canny(blur_frame, 50, 150)


# In[5]:


def preprocess(frame):
    gray = to_grayscale(frame)
    blur = apply_blur(gray)
    edges = detect_edges(blur)

    # Returns all three processed images
    return gray, blur, edges


# In[6]:


import time


# In[7]:


def calculate_fps(time_diffs):
    np_array = np.array(time_diffs)
    mean_val = np_array.mean()
    return 1/mean_val


# In[8]:


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


# In[9]:


def combine_frames(frame, edges):
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.hconcat([frame, edge_bgr])


# In[12]:


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

    time_diffs = []
    start_time = time.time()

    while True:
    # ret: 프레임을 성공적으로 읽었으면 True, 아니면 False
    # frame: 읽어온 1개의 이미지 프레임 (NumPy 배열 형식)
        ret, frame = cap.read()

        # 
        if not ret:
            print("프레임을 가져올 수 없습니다. 종료합니다.")
            break
        orig_f = frame.copy()
        end_time = time.time()
        time_diffs.append(end_time - start_time)
        if len(time_diffs) == 30:
            fps = calculate_fps(time_diffs)

            time_diffs.pop(0)
            draw_fps(frame, fps)

        start_time = end_time

        gray_f, blur_f, edges_f = preprocess(orig_f)

        combined = combine_frames(frame, edges_f)

        # 화면에 프레임 표시
        cv2.imshow('Combined', combined)

        # 4. Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[13]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




