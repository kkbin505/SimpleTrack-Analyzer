import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import time

# 配置文件与设备
VIDEO_PATH = "ST_20260103_1613_Tianma_Render.mp4" # 请确认你的视频路径

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = torch.device("cuda")

ROAD_CLASS_ID = 0 # Cityscapes 数据集中道路的类别ID

# 1. 加载模型
print(f"正在加载 SegFormer B2 模型至 {DEVICE}...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(DEVICE)
model.eval()
# 【重点】RTX 2060 开启半精度推理，速度直接翻倍！
model.half() 

def draw_middle_third_boundaries(display_frame, road_mask):
    h, w = road_mask.shape[:2]
    
    # 1. 设定纵向黄金区间
    y_start = int(h * 0.54)
    y_end = int(h * 0.62)
    
    # 2. 创建一个只包含该区域的裁剪 Mask
    # 将区间外的 Mask 全部涂黑
    roi_mask = np.zeros_like(road_mask)
    roi_mask[y_start:y_end, :] = road_mask[y_start:y_end, :]
    
    # 3. 提取轮廓 (只处理中间这一横条)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return display_frame
    
    # 只取面积最大的路面轮廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 4. 提取左右边界点
    left_side = {}
    right_side = {}
    
    for pt in main_contour:
        x, y = pt[0][0], pt[0][1]
        # 只记录在该区间内的点
        if y_start <= y <= y_end:
            if y not in left_side or x < left_side[y]:
                left_side[y] = x
            if y not in right_side or x > right_side[y]:
                right_side[y] = x
                
    # 5. 渲染边界
    # 绘制左侧 (红色)
    for y, x in left_side.items():
        cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)
    
    # 绘制右侧 (绿色)
    for y, x in right_side.items():
        cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)

    # 可选：画出 ROI 的分界线方便观察
    cv2.line(display_frame, (0, y_start), (w, y_start), (255, 255, 255), 1)
    cv2.line(display_frame, (0, y_end), (w, y_end), (255, 255, 255), 1)
        
    return display_frame

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


# ====== 1. 你的最新坐标与物理参数 ======
# 坐标顺序：左下, 左上, 右上, 右下
# SRC_PTS = np.float32([[695, 709], [821, 640], [981, 643], [1110, 719]])
SRC_PTS = np.float32([[704, 704], [844, 623], [978, 627], [1097, 699]])

# 设定 BEV 窗口大小：宽 400 (对应 3.43m), 高 800 (对应纵向约 10-15m)
BEV_W, BEV_H = 1920, 800
# DST_PTS = np.float32([[0, BEV_H], [0, 0], [BEV_W, 0], [BEV_W, BEV_H]])
ref_pix_width = 200
MARGIN = (BEV_W - ref_pix_width) // 2 # 
DST_PTS = np.float32([
    [MARGIN, BEV_H],                # 左下 -> x=300
    [MARGIN, 0],                    # 左上 -> x=300
    [MARGIN + ref_pix_width, 0],    # 右上 -> x=500
    [MARGIN + ref_pix_width, BEV_H] # 右下 -> x=500
])

# 物理比例尺
XM_PER_PIX = 3.43 / BEV_W  
YM_PER_PIX = 10.0 / BEV_H   # 这里的 10.0 是估算的纵向长度，后续可以微调

def process_bev_step(frame, road_mask, M, bev_size=(BEV_W, BEV_H)):
    bev_view = cv2.warpPerspective(frame, M, bev_size)
    bev_mask = cv2.warpPerspective(
        road_mask,
        M,
        bev_size,
        flags=cv2.INTER_NEAREST
    )

    _, bev_mask = cv2.threshold(bev_mask, 1, 255, cv2.THRESH_BINARY)
    return bev_view, bev_mask

def overlay_transparent_mask(background_img, mask_img, color=(0, 255, 0), alpha=0.4):
    """
    将黑白 Mask 以半透明色块的形式叠加到背景图像上。
    
    Args:
        background_img (numpy.ndarray): 原始背景图像 (BGR 彩色)
        mask_img (numpy.ndarray): 黑白 Mask 图像 (灰度图, 0为背景, 255为目标)
        color (tuple): 叠加色块的颜色，格式为 BGR，默认为绿色 (0, 255, 0)
        alpha (float): 透明度系数，范围 0.0 - 1.0。越小越透明，越大颜色越深。默认 0.4
        
    Returns:
        numpy.ndarray: 叠加完成后的彩色图像
    """
    # 1. 创建一个和背景一样大小的全黑彩色画布
    h, w, c = background_img.shape
    colored_mask = np.zeros((h, w, c), dtype=np.uint8)
    
    # 2. 利用 numpy 的布尔索引，将 Mask 中为白色的区域填上指定的颜色
    # mask_img == 255 找到了所有路面像素的位置
    colored_mask[mask_img == 255] = color
    
    # 3. 核心步骤：加权混合 (Blending)
    # 公式: 输出 = 背景图 * beta + 前景图 * alpha + gamma
    # 这里我们让背景图保持原样 (权重 1.0)，叠加图层给予一定的透明度权重 (alpha)
    blended_image = cv2.addWeighted(background_img, 1.0, colored_mask, alpha, 0)
    
    return blended_image

class TrackMemory:
    def __init__(self, alpha=0.15):
        """
        alpha: 记忆权重 (0.0 ~ 1.0)。
        值越小，行车线越稳（记忆力强）；值越大，行车线对当前帧反应越快。
        """
        self.alpha = alpha
        self.prev_coeffs = None  # 存储上一帧的多项式系数 [A, B, C]

    def smooth_fit(self, bev_mask):
        """
        在 BEV 空间提取中心线并进行平滑拟合
        """
        h, w = bev_mask.shape
        # 采样点：从底部到顶部生成 30 个纵向坐标点
        plot_y = np.linspace(0, h - 1, 30)
        
        # 1. 横向扫描提取中点
        center_pts = []
        for y in range(h - 1, 0, -20):  # 每隔 20 像素扫描一行
            pixels = np.where(bev_mask[y, :] == 255)[0]
            if len(pixels) > 40:  # 确保检测到的路面宽度合理
                mid_x = (pixels[0] + pixels[-1]) / 2
                center_pts.append([mid_x, y])
        
        if len(center_pts) < 8:  # 如果点太少，说明这一帧识别质量太差，跳过
            return None
        
        pts = np.array(center_pts)
        
        # 2. 二阶多项式拟合: x = Ay^2 + By + C
        try:
            current_coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
        except:
            return None

        # 3. 执行指数移动平均 (EMA) 平滑
        if self.prev_coeffs is None:
            self.prev_coeffs = current_coeffs
        else:
            # 核心记忆逻辑：当前系数与历史系数按比例融合
            self.prev_coeffs = self.alpha * current_coeffs + (1 - self.alpha) * self.prev_coeffs
        
        # 4. 根据平滑后的系数计算路径点坐标
        fit_x = self.prev_coeffs[0] * plot_y**2 + self.prev_coeffs[1] * plot_y + self.prev_coeffs[2]
        smooth_bev_pts = np.column_stack((fit_x, plot_y)).astype(np.float32)
        
        return smooth_bev_pts

    def project_to_fpv(self, bev_pts, M_inv):
        """
        利用逆矩阵将 BEV 坐标反向投影回第一视角 (FPV)
        """
        if bev_pts is None:
            return None
        # 调整形状以匹配 cv2.perspectiveTransform 的输入要求 (N, 1, 2)
        pts_reshaped = bev_pts.reshape(-1, 1, 2)
        # 执行逆透视变换
        fpv_pts = cv2.perspectiveTransform(pts_reshaped, M_inv)
        return fpv_pts.reshape(-1, 2).astype(np.int32)

# 假设你已经定义了之前的 TrackMemory 类
# track_mem = TrackMemory(alpha=0.15) 

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {VIDEO_PATH}")
        return
    
    prev_time = 0
    # 初始化记忆模块
    track_mem = TrackMemory(alpha=0.15) 
    
    # 计算矩阵及其逆矩阵
    M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    M_inv = np.linalg.inv(M)
    
    XM_PER_PIX = 3.43 / BEV_W
    YM_PER_PIX = XM_PER_PIX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 去畸变
        frame_undistorted = gopro_16_9_simple_undistort(frame)
        h, w = frame_undistorted.shape[:2]
        
        # 2. 设定 ROI 区域 (1080p 原始尺度)
        y_start, y_end = int(h * 0.54), int(h * 0.62)
        roi_frame = frame_undistorted[y_start:y_end, :]
        roi_h, roi_w = roi_frame.shape[:2] 

        # ==========================================
        # 【核心优化 1】推理前分辨率减半 (提速 3-4 倍)
        # ==========================================
        roi_small = cv2.resize(roi_frame, (roi_w // 2, roi_h // 2))
        roi_pil = Image.fromarray(cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB))
        
        # 搬运到 GPU 并开启半精度
        inputs = processor(images=roi_pil, return_tensors="pt").to(DEVICE)
        if DEVICE.type == 'cuda':
            inputs = {k: v.half() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            
            # ==========================================
            # 【核心优化 2】GPU 直接插值还原回 1080p ROI 尺寸
            # ==========================================
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits, size=(roi_h, roi_w), mode="bilinear", align_corners=False
            )
            pred_seg_roi = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # 3. 提取局部道路 Mask 并贴回全图 (保证 IPM 坐标对齐)
        roi_road_mask = (pred_seg_roi == ROAD_CLASS_ID).astype(np.uint8) * 255
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[y_start:y_end, :] = roi_road_mask

        # 4. 执行 IPM 投影
        bev_frame_color = cv2.warpPerspective(frame_undistorted, M, (BEV_W, BEV_H))
        bev_mask = cv2.warpPerspective(road_mask, M, (BEV_W, BEV_H))

        # ==========================================
        # 【核心优化 3】引入时间平滑记忆与行车线投影
        # ==========================================
        smooth_bev_path = track_mem.smooth_fit(bev_mask)
        fpv_line = track_mem.project_to_fpv(smooth_bev_path, M_inv)

        # 5. 渲染第一视角结果 (display_frame)
        overlay_result = overlay_transparent_mask(frame_undistorted, road_mask, color=(0, 255, 0), alpha=0.4)
        display_frame = overlay_result.copy()

        # 绘制投影回来的平滑行车线 (黄色)
        # if fpv_line is not None and len(fpv_line) > 1:
        #     cv2.polylines(display_frame, [fpv_line], False, (0, 255, 255), 3, cv2.LINE_AA)

        # 6. 渲染上帝视角结果 (bev_with_overlay)
        bev_overlay_result = overlay_transparent_mask(bev_frame_color, bev_mask, color=(0, 255, 0), alpha=0.4)
        bev_with_overlay = bev_overlay_result.copy()
        
        # 在 BEV 上也画出平滑路径
        if smooth_bev_path is not None:
            cv2.polylines(bev_with_overlay, [smooth_bev_path.astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)

        # 7. 绘制 UI 元素与诊断信息
        pts_int = SRC_PTS.astype(np.int32)
        for pt in pts_int: cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
        cv2.line(display_frame, (0, y_start), (w, y_start), (255, 255, 255), 1)
        cv2.line(display_frame, (0, y_end), (w, y_end), (255, 255, 255), 1)
        
        # 计算并显示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 8. 窗口显示 (缩小 BEV 提升渲染效率)
        cv2.imshow("SegFormer View (FPV)", display_frame)
        
        display_scale = 0.5
        bev_small = cv2.resize(bev_with_overlay, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("BEV with Road Overlay", bev_small)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()