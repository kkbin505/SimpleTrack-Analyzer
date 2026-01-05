

import cv2
import numpy as np

# 存储选取的点
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(frame_undistorted, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", frame_undistorted)
        if len(points) == 4:
            print("选取的四点坐标：", points)

def gopro_16_9_simple_undistort(frame):
    h, w = frame.shape[:2]
    # 针对 16:9 比例微调的通用 K 矩阵
    K = np.array([
        [w * 0.45, 0, w * 0.5],
        [0, h * 0.78, h * 0.5],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 针对标准镜头而非 Max Wide 的畸变系数
    # k1, k2, k3, k4
    D = np.array([-0.02, 0.01, -0.01, 0.005], dtype=np.float32)
    
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


# 使用示例
# map1, map2 = gopro_mini_undistort_init(1920, 1080)
# frame_undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

# 加载视频第一帧
cap = cv2.VideoCapture("tianma_ipm.mp4")
ret, img = cap.read()
cap.release()

frame_undistorted = gopro_16_9_simple_undistort(img)


cv2.imshow("Calibration", frame_undistorted)
cv2.setMouseCallback("Calibration", click_event)
print("请在路面上由近及远、顺时针点击4个点（形成一个梯形）")
cv2.waitKey(0)
cv2.destroyAllWindows()