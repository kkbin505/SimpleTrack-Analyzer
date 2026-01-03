# SimpleTrack-Analyzer
Gopro track data overlay

![alt text](img/image.png)

## Key Features:


## Precision G-Force Visualization:

Friction Circle Analysis: A smooth, anti-aliased G-force meter designed to help drivers visualize Trail Braking and tire grip limits.

Dynamic Motion Trails: Tracks the history of your G-loads to identify "cross-shaped" inputs vs. professional "arc-shaped" transitions.

## Driver-Centric Logic:

Rolling Mean Smoothing: Filtered IMU data to eliminate GoPro sensor noise while preserving transient response.


Inversion Support: One-click 180° rotation for inverted cockpit camera mounts.

Create conda：

```bash

# Create a new conda environment
conda create -n track_analysis_env python=3.10 -y 

# Activate the environment
conda activate track_analysis_env

# Install dependencies (Note: numpy < 2 is required for compatibility)
pip install "numpy<2" pandas opencv-python telemetry-parser matplotlib PyQt6 Pillow

```

run:

```bash
python gopro_overlay.py

```

## 📖 Background

Since **GoPro Quik for Desktop** is no longer being updated for professional telemetry overlays, many users (including myself) found it difficult to find a lightweight, high-performance tool to visualize track data. 

I created **SimpleTrack-Analyzer** to solve this—providing a simple, efficient way to overlay G-Force and IMU data.




