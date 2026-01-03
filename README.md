# SimpleTrack-Analyzer
Gopro track data overlay


Key Features:


# Precision G-Force Visualization:

Friction Circle Analysis: A smooth, anti-aliased G-force meter designed to help drivers visualize Trail Braking and tire grip limits.

Dynamic Motion Trails: Tracks the history of your G-loads to identify "cross-shaped" inputs vs. professional "arc-shaped" transitions.

# Driver-Centric Logic:

Rolling Mean Smoothing: Filtered IMU data to eliminate GoPro sensor noise while preserving transient response.

Smart Seek: Instant preview with automatic trail resetting when scrubbing through the timeline.

Inversion Support: One-click 180° rotation for inverted cockpit camera mounts.


Create conda：

conda create -n track_analysis_env python=3.10 -y 

conda info --envs

conda activate racing_pilot_env

pip install "numpy<2" pandas opencv-python telemetry-parser matplotlib PyQt6

run:

python gopro_overlay.py






