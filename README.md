- 👋 Hi, I’m @yusic-1
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

import cv2
import numpy as np

# 指定外接攝像頭的索引
external_camera_index = 1

# 初始化外接攝像頭
cap = cv2.VideoCapture(external_camera_index)

# 初始化背景分割器
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # 讀取一幀的影像
    ret, frame = cap.read()

    # 背景差異法進行移動檢測
    fg_mask = bg_subtractor.apply(frame)

    # 使用形態學運算進行前景物體的去噪
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))

    # 進行輪廓檢測
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 繪製框選框
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 忽略小區域的雜訊
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示影像窗口
    cv2.imshow('Motion Detection', frame)

    # 按下 'q' 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
