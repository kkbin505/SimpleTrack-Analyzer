import pandas as pd
import matplotlib.pyplot as plt
import ast # 用于安全地将字符串 "[1, 2, 3]" 转为列表

def plot_gopro_vectors(csv_file):
    # 1. 加载数据
    df = pd.read_csv(csv_file)

    # 2. 拆解 accl 向量
    # 使用 ast.literal_eval 处理字符串，并展开成三列
    accel_split = df['accl'].apply(ast.literal_eval).apply(pd.Series)
    accel_split.columns = ['Ax', 'Ay', 'Az'] # X:垂直, Y:横向, Z:前后

    # 3. 数据平滑 (赛道分析必备)
    window = 20
    df['Ax_smooth'] = accel_split['Ax'].rolling(window=window).mean()
    df['Ay_smooth'] = accel_split['Ay'].rolling(window=window).mean()
    df['Az_smooth'] = accel_split['Az'].rolling(window=window).mean()


    df['AF_A_G'] = df['Az_smooth'] / 9.81
    df['ALateral_G'] = df['Ay_smooth'] / 9.81
    df['AVertical_G'] = df['Ax_smooth'] / 9.81

    # 4. 绘图
    plt.figure(figsize=(12, 7))
    
    # Ax: 刹车/油门 (刹车通常是大幅度的负值或正值，取决于安装方向)
    plt.plot(df['timestamp_ms'], df['AF_A_G'], label='Longitudinal (Ax) - Brake/Throttle', color='red')
    
    # Ay: 横向加速度 (转弯时的拉力)
    plt.plot(df['timestamp_ms'], df['ALateral_G'], label='Lateral (Ay) - Cornering', color='blue')
    
    plt.plot(df['timestamp_ms'], df['AVertical_G'], label='Vertical (Az) - Cornering', color='green')

    plt.title('GoPro Telemetry: Acceleration Vector Analysis')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Acceleration (m/s^2 or G)')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    csv_name = "demo\Tianma_imu.csv" 
    plot_gopro_vectors(csv_name) 
    
