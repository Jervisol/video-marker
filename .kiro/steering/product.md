# Product Overview

Real-time video object detection and annotation system (视频实时标记系统) that allows users to upload videos, select object classes to detect, and view detection results in real-time or export annotated videos.

## Core Features

- **Video Upload**: Support for mp4, avi, mov, mkv formats
- **Real-time Detection**: WebSocket-based frame-by-frame object detection with live canvas overlay
- **Multi-Model Support**: YOLO11n, YOLOv8n, EfficientDet, and custom models
- **Class Selection**: Dynamic UI for selecting which object classes to detect
- **Detection Timeline**: Visual timeline showing when objects appear in the video
- **Video Export**: Background task processing to generate annotated videos with adjustable FPS
- **Click Detection**: Interactive feature to identify objects by clicking on video frames

## User Workflow

1. Upload video file through web interface
2. Select detection model (YOLO11n, YOLOv8n, etc.)
3. Choose object classes to detect from available categories
4. Start real-time detection or export annotated video
5. View results with bounding boxes, labels, and confidence scores
