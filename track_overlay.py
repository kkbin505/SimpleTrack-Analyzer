import cv2
import pandas as pd
import numpy as np
import ast
from collections import deque

class SimpleTrackOverlay:
    def __init__(self, video_path, gps_csv, imu_csv):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 数据加载与清洗
        self.trail_length = int(self.fps * 2.0) 
        self.pts = deque(maxlen=self.trail_length)
        self.df_imu = self.process_imu(imu_csv)
        print(f"🎬 视频加载成功: {self.w}x{self.h} @ {self.fps}fps")

    def process_imu(self, path):
        df = pd.read_csv(path)
        # 拆解 GoPro accl 向量 [x, y, z]
        accel = df['accl'].apply(ast.literal_eval).apply(pd.Series)

        accel = accel.rolling(window=40, min_periods=1, center=True).mean()
        # 赛化缤越校准：根据倒挂情况，Ax/Ay 可能需要调整正负号
        # offset_ax = accel[0].iloc[:100].mean() 
        df['ax'] = (accel[2]) / 9.80665
        df['ay'] = -accel[1] / 9.80665
        
        # 强制转换时间戳为数值，防止索引匹配报错
        df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
        return df.dropna(subset=['timestamp_ms'])

    def draw_telemetry(self, frame, imu_row):
        # 1. 核心修复：先翻转画面（抵消 GoPro 倒挂）
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # 2. 绘制 G-G Diagram (右下角)
        center = (self.w - 200, self.h - 200)
        radius = 120
        # 绘制背景圆
        cv2.circle(frame, center, radius, (200, 200, 200), 2) # 1.0G 线
        cv2.circle(frame, center, radius // 2, (100, 100, 100), 1) # 0.5G 线
        
        # 绘制实时红点 (根据你的物理定义映射)
        dot_x = int(center[0] + imu_row['ay'] * radius)
        dot_y = int(center[1] - imu_row['ax'] * radius)
        self.pts.appendleft((dot_x, dot_y))

        # 3. 绘制渐隐轨迹 (核心视觉效果)
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
                
            # 计算粗细和透明度：索引越大(i)，点越旧，越细越淡
            thickness = int(np.sqrt(self.trail_length / float(i + 1)) * 2.5)
            # 颜色从红色渐变为深红/黑色 (0, 0, 255 -> 0, 0, 50)
            alpha = float(len(self.pts) - i) / len(self.pts)
            color = (0, 0, int(255 * alpha))
            
            cv2.line(frame, self.pts[i - 1], self.pts[i], color, thickness)
        
        # 4. 绘制最前端的当前实时点
        cv2.circle(frame, (dot_x, dot_y), 10, (255, 255, 255), -1) # 白色外圈
        cv2.circle(frame, (dot_x, dot_y), 7, (0, 0, 255), -1)   # 红色中心
        
        
        # 3. 添加时间水印，方便核对区间
        timestamp_str = f"G-G Segment: {imu_row['timestamp_ms']/1000:.1f}s"
        cv2.putText(frame, timestamp_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def run(self, output_file="SimpleTrack_Segment.mp4", start_min=2, end_min=4):
        # 计算起始和结束帧
        start_frame = int(start_min * 60 * self.fps)
        end_frame = int(end_min * 60 * self.fps)
        
        # 将视频指针快速定位到起始时间
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.w, self.h))
        
        print(f"🚀 开始渲染特定区间: {start_min}min -> {end_min}min")
        
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 计算当前绝对毫秒时间戳进行对齐
            ts = (frame_idx / self.fps) * 1000
            
            # 在 IMU 表中找最接近的时间点
            idx_imu = (self.df_imu['timestamp_ms'] - ts).abs().idxmin()
            
            # 渲染 UI 并写入
            frame = self.draw_telemetry(frame, self.df_imu.iloc[idx_imu])
            out.write(frame)
            
            frame_idx += 1
            if frame_idx % 300 == 0:
                progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
                print(f"⏳ 进度: {progress:.1f}% | 当前时间: {frame_idx // self.fps}秒")

        self.cap.release()
        out.release()
        print(f"✅ 区间渲染完成！保存至: {output_file}")

# 启动！
video_path = "demo/Tianma.MP4"
csv_gps = "demo/Tianma_gps.csv" # 暂时占位，逻辑中未启用
csv_imu = "demo/Tianma_imu.csv"
output_path = "demo/Tianma_2-4min_Overlay.mp4"

overlay = SimpleTrackOverlay(video_path, csv_gps, csv_imu)
# 调用 run 时指定 3 分钟到 4 分钟
overlay.run(output_file=output_path, start_min=3, end_min=4)