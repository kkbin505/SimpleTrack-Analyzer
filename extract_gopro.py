import pandas as pd
from telemetry_parser import Parser
import os

def extract_gopro_data(video_path):
    print(f"🚀 正在解析视频: {video_path}...")
    tp = Parser(video_path)
    
    # 1. 抓取所有原始数据流
    all_streams = tp.telemetry()
    
    gps_list = []
    
    # 2. 深度挖掘 'Default' 里的 GPS 数据
    if 'Default' in all_streams:
        print("🔍 正在扫描 Default 流中的 GPS 信号...")
        for entry in all_streams['Default']:
            # 检查是否为包含 GPS 信息的字典
            if isinstance(entry, dict) and 'Name' in entry:
                # Hero 11 的 GPS 标识通常包含这个字符串
                if "GPS (Lat., Long., Alt., Speed, 2D speed)" in entry['Name']:
                    # 获取该包的基础微秒时间戳
                    base_ts_us = entry.get('TimestampUs', 0)
                    
                    # 提取 val 中的具体点（val 通常是一个列表的列表）
                    if 'val' in entry and isinstance(entry['val'], list):
                        for point in entry['val']:
                            # Hero 11 的 val 格式通常是 [lat, lon, alt, speed_2d, speed_3d]
                            if len(point) >= 4:
                                gps_list.append({
                                    'timestamp_ms': base_ts_us / 1000.0,
                                    'lat': point[0],
                                    'lon': point[1],
                                    'speed': point[3]  # 地面 2D 速度
                                })

    # 3. 构造真正的 GPS DataFrame
    df_gps_final = pd.DataFrame(gps_list)

    # 4. 提取 IMU (通常比较标准)
    df_imu = pd.DataFrame(tp.normalized_imu())
    # 确保 IMU 也有时间戳列
    if 'cts' in df_imu.columns:
        df_imu.rename(columns={'cts': 'timestamp_ms'}, inplace=True)
    elif 'TimestampUs' in df_imu.columns:
        df_imu['timestamp_ms'] = df_imu['TimestampUs'] / 1000.0

    # 5. 保存
    base_name = os.path.splitext(video_path)[0]
    gps_file = f"{base_name}_gps.csv"
    imu_file = f"{base_name}_imu.csv"

    if not df_gps_final.empty:
        df_gps_final.to_csv(gps_file, index=False)
        print(f"✅ GPS 提取成功：共 {len(df_gps_final)} 条高频位置点")
    else:
        print("❌ 警告：未在 Default 流中匹配到 GPS5 数据，请检查相机是否开启了 GPS 记录。")

    df_imu.to_csv(imu_file, index=False)
    print(f"🧪 IMU 提取成功：共 {len(df_imu)} 条")

if __name__ == "__main__":
    video_file = r"D:\LiZhen\Github\racing_pilot\demo\Tianma.MP4"
    extract_gopro_data(video_file)