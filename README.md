# Project Overview

This platform integrates radar point cloud data and Kinect-based skeleton tracking.

## Folder Structure

- **Calibration Data**: Contains multiple sets of central 3D positions obtained from Kinect for calibration purposes.
- **Dataset**: The final preprocessed dataset, ready for deep learning applications.
- **Examples**: Sample code sourced from the Azure Kinect repository, used for recording and computing skeleton graphs via the Kinect SDK.
- **Process**: Includes core functionalities, models, and utility functions for data processing.

## How to Build Your Own Dataset

To generate a custom dataset, follow the workflow in `main.py`, starting with MKV files.
