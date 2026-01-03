import sys
import time  
import cv2
import pandas as pd
import numpy as np
import os
import subprocess
from collections import deque
from telemetry_parser import Parser
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QMessageBox, QCheckBox, QStyleOptionSlider, QStyle)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QRect
from PIL import Image, ImageDraw, ImageFont

# --- Utility: Format Time ---
def format_time(frames, fps):
    if fps <= 0: return "00:00.00"
    total_seconds = frames / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

# --- Custom Slider with Range Highlighting ---
class RangeSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.in_pos = 0
        self.out_pos = 0

    def set_range_marks(self, in_f, out_f):
        self.in_pos = in_f
        self.out_pos = out_f
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        gr = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self)
        
        if self.maximum() > 0:
            s_width = gr.width()
            x_in = gr.left() + (self.in_pos / self.maximum()) * s_width
            x_out = gr.left() + (self.out_pos / self.maximum()) * s_width
            
            # Slider background (Dark Grey)
            painter.setBrush(QColor(60, 60, 60))
            painter.drawRect(gr)
            # Selected Range (Racing Orange)
            painter.setBrush(QColor(230, 126, 34))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRect(int(x_in), gr.top(), int(x_out - x_in), gr.height()))

        super().paintEvent(event)

# --- Core Engine ---
class GoProDirectEngine:
    def reset_trail(self):
        """Clear G-force trail to prevent lines crossing the screen on jump"""
        self.pts.clear()

    def __init__(self, video_path, rotate_180=True):
        self.video_path = video_path
        self.rotate_180 = rotate_180
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 1. Preload Roboto-Bold font
        try:
            self.font_path = "font\\static\\Roboto-Bold.ttf"
            self.font_main = ImageFont.truetype(self.font_path, 28)
            print(f"📖 Font loaded successfully: {self.font_path}")
        except:
            print("⚠️ Roboto-Bold.ttf not found, using system default font")
            self.font_main = ImageFont.load_default()
        
        # Load data and check GPS
        self.df_imu, self.has_gps = self.extract_data_from_mp4(video_path)
        self.pts = deque(maxlen=int(self.fps * 2.0))

    def extract_data_from_mp4(self, path):
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"📂 Loading video file: {path} ({file_size:.1f} MB)")
        
        # 1. Visual progress simulation
        print(f"🚀 Parsing metadata streams...")
        for i in range(1, 41):
            time.sleep(0.01) # Simulate quick scan
            print(f"\r🔍 Scanning tracks: [{int(i)*'=':40s}] {i*2.5:4.1f}%", end="")
        
        # Actual parser call
        tp = Parser(path)
        all_streams = tp.telemetry()
            
        # Extract and smooth data
        df = pd.DataFrame(tp.normalized_imu())
        if 'cts' in df.columns:
            df['timestamp_ms'] = df['cts']
        
        accel = df['accl'].apply(pd.Series).rolling(window=50, min_periods=1, center=True).mean()
        df['ax'] = (accel[2]) / 9.80665
        df['ay'] = -accel[1] / 9.80665
        
        # Check GPS status
        gps_found = False
        if isinstance(all_streams, list):
            gps_found = any("GPS" in str(s.get('Name', '')) for s in all_streams)
        elif isinstance(all_streams, dict):
            gps_found = any("GPS" in str(k) for k in all_streams.keys())
        
        print(f"\n✅ Parsing complete! GPS: {'Found' if gps_found else 'Not Found'}\n")
        return df.dropna(subset=['timestamp_ms']), gps_found

    def draw_frame(self, frame, ts_ms, draw_g=True, rotate = True):
        if rotate: 
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not draw_g:
            return frame
        
        idx = (self.df_imu['timestamp_ms'] - ts_ms).abs().idxmin()
        row = self.df_imu.iloc[idx]
        center, radius = (self.w - 200, self.h - 200), 120
        cv2.circle(frame, center, radius, (200, 200, 200), 2) # 1.0G Line
        cv2.circle(frame, center, radius // 2, (100, 100, 100), 1) # 0.5G Line
        
        # Crosshair lines
        cv2.line(frame, (center[0] - 100, center[1]), (center[0] + 100, center[1]),(255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (center[0], center[1] - 100), (center[0], center[1] + 100), (255, 255, 255), 1, cv2.LINE_AA)
        ax, ay = row['ax'], row['ay']

        # Switch to PIL for high-quality text rendering
        roi_size = 500
        sub_frame = frame[self.h - roi_size:, self.w - roi_size:]
        img_pil = Image.fromarray(cv2.cvtColor(sub_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        local_center = (250, 250)

        # Prepare Text
        txt_brake = f"{max(0, ax):.2f}G BRAKE"
        txt_accel = f"{abs(min(0, ax)):.2f}G ACCEL"
        txt_lat = f"{abs(ay):.2f}G LAT"

        # Top: BRAKE
        draw.text((local_center[0] - 30, local_center[1] - 100 ), txt_brake, font=self.font_main, fill=(255, 255, 255))
        # Bottom: ACCEL
        draw.text((local_center[0] - 30, local_center[1] + 100 + 70 ), txt_accel, font=self.font_main, fill=(255, 255, 255))
        # Side: LAT
        draw.text((local_center[0] - 210 , local_center[1] + 30 ), txt_lat, font=self.font_main, fill=(255, 255, 255))

        sub_frame_processed = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        frame[self.h - roi_size:, self.w - roi_size:] = sub_frame_processed

        # G-Force Dot and Trail
        dot = (int(center[0] + row['ay'] * radius), int(center[1] - row['ax'] * radius))
        self.pts.appendleft(dot)
        for i in range(1, len(self.pts)):
            thick = int(np.sqrt(len(self.pts)/(i+1))*2.5)
            color = (0, 0, int(255 * (len(self.pts)-i)/len(self.pts)))
            cv2.line(frame, self.pts[i-1], self.pts[i], color, thick)
        cv2.circle(frame, dot, 10, (255, 255, 255), -1)
        cv2.circle(frame, dot, 7, (0, 0, 255), -1)

        return frame

# --- Main Application Window ---
class TrackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SimpleTrack Analyzer")
        self.engine = None
        self.in_f, self.out_f = 0, 0
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.preview = QLabel("Please Load GoPro Video")
        self.preview.setFixedSize(800, 450)
        self.preview.setStyleSheet("background: black; border-radius: 5px;")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview)

        self.time_lbl = QLabel("In: 00:00.00 | Out: 00:00.00 | Cur: 00:00.00")
        self.time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_lbl.setStyleSheet("font-family: Consolas; font-size: 13px;")
        layout.addWidget(self.time_lbl)

        self.slider = RangeSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.update_preview)
        layout.addWidget(self.slider)

        opts = QHBoxLayout()
        self.cb_gps = QCheckBox("Render GPS")
        self.cb_g = QCheckBox("Render G-Force Circle"); self.cb_g.setChecked(True)
        self.cb_r = QCheckBox("Rotate 180°"); self.cb_r.setChecked(True)
        self.cb_a = QCheckBox("Keep Original Audio"); self.cb_a.setChecked(True)
        for c in [self.cb_gps, self.cb_g, self.cb_r, self.cb_a]: opts.addWidget(c)
        layout.addLayout(opts)

        btns = QHBoxLayout()
        b_load = QPushButton("📂 Load Video"); b_load.clicked.connect(self.load_video)
        b_in = QPushButton("Set In"); b_in.clicked.connect(self.set_in)
        b_out = QPushButton("Set Out"); b_out.clicked.connect(self.set_out)
        self.b_run = QPushButton("Export Video"); self.b_run.clicked.connect(self.render_video)
        self.b_run.setStyleSheet("background: #27ae60; color: white; font-weight: bold;")
        for b in [b_load, b_in, b_out, self.b_run]: btns.addWidget(b)
        layout.addLayout(btns)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if path:
            self.engine = GoProDirectEngine(path, self.cb_r.isChecked())
            self.out_f = int(self.engine.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            self.slider.setMaximum(self.out_f)
            
            # --- GPS Gray-out Logic ---
            if not self.engine.has_gps:
                self.cb_gps.setChecked(False)
                self.cb_gps.setEnabled(False)
                self.cb_gps.setText("GPS (No Signal Detected)")
            else:
                self.cb_gps.setEnabled(True)
                self.cb_gps.setChecked(True)
                self.cb_gps.setText("Render GPS")

            self.slider.set_range_marks(0, self.out_f)
            self.update_preview(0)

    def update_preview(self, pos):
        if not self.engine: return
        
        self.engine.reset_trail()
        t_in = format_time(self.in_f, self.engine.fps)
        t_out = format_time(self.out_f, self.engine.fps)
        t_cur = format_time(pos, self.engine.fps)
        
        self.time_lbl.setText(f"In: {t_in} | Out: {t_out} | Cur: {t_cur}")
        
        self.engine.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = self.engine.cap.read()
        is_rotate = self.cb_r.isChecked()
        if ret:
            frame = self.engine.draw_frame(frame, (pos/self.engine.fps)*1000, self.cb_g.isChecked(), rotate=is_rotate)
            frame = cv2.resize(frame, (800, 450))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, 800, 450, QImage.Format.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(img))

    def set_in(self): self.in_f = self.slider.value(); self.slider.set_range_marks(self.in_f, self.out_f)
    def set_out(self): self.out_f = self.slider.value(); self.slider.set_range_marks(self.in_f, self.out_f)

    def render_video(self):
        if not self.engine: return
        is_rotate = self.cb_r.isChecked()
        temp_v, final_v = "temp_silent.mp4", "SimpleTrack_Render.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_v, fourcc, self.engine.fps, (self.engine.w, self.engine.h))
        
        self.engine.cap.set(cv2.CAP_PROP_POS_FRAMES, self.in_f)
        total_frames = self.out_f - self.in_f
        
        print(f"\n🎬 Starting Render Job...")
        print(f"Interval: {format_time(self.in_f, self.engine.fps)} -> {format_time(self.out_f, self.engine.fps)}")

        for i in range(self.in_f, self.out_f):
            ret, frame = self.engine.cap.read()
            if not ret: break
            
            frame = self.engine.draw_frame(frame, (i/self.engine.fps)*1000, self.cb_g.isChecked(), rotate=is_rotate)
            out.write(frame)
            
            # Progress bar logic
            if i % 5 == 0:
                processed = i - self.in_f
                progress = (processed / total_frames) * 100
                cur_time = format_time(i, self.engine.fps)
                print(f"\r⏳ Progress: [{int(progress/2)*'=':50s}] {progress:4.1f}% | Position: {cur_time}", end="")

        out.release()

        if self.cb_a.isChecked():
            start_s, dur_s = self.in_f/self.engine.fps, (self.out_f-self.in_f)/self.engine.fps
            cmd = ['ffmpeg', '-y', '-i', temp_v, '-ss', str(start_s), '-t', str(dur_s), '-i', self.engine.video_path,
                   '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', final_v]
            subprocess.run(cmd, shell=True)
            if os.path.exists(temp_v): os.remove(temp_v)
        else:
            if os.path.exists(final_v): os.remove(final_v)
            os.rename(temp_v, final_v)
        
        print(f"\n🏁 Task Finished. Saved to: {final_v}")
        QMessageBox.information(self, "Finished", f"Video exported successfully: {final_v}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrackApp(); window.show()
    sys.exit(app.exec())