# Technology Stack

## Backend

- **Framework**: FastAPI (async web framework)
- **Object Detection**: Ultralytics YOLO (YOLOv8n, YOLO11n models)
- **Computer Vision**: OpenCV (cv2) for video processing
- **Image Processing**: PIL/Pillow, NumPy
- **Logging**: Loguru
- **Concurrency**: ThreadPoolExecutor for CPU-intensive tasks

## Frontend

- **Template Engine**: Jinja2
- **Real-time Communication**: Native WebSocket API
- **Visualization**: Chart.js for detection timeline
- **Styling**: Vanilla CSS with responsive grid layouts

## Architecture Patterns

- **Async/Await**: FastAPI endpoints use async for I/O operations
- **Thread Pool**: Model inference runs in separate threads to avoid blocking
- **WebSocket Manager**: Connection manager pattern for handling multiple clients
- **Background Tasks**: Export feature uses ThreadPoolExecutor for long-running video processing
- **Task Management**: ExportTaskManager with thread-safe operations using locks

## Common Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Development Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Run Production Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Configuration

- Config file: `config.py`
- Upload folder: `static/uploads`
- Default model: YOLO11n
- Allowed video extensions: mp4, avi, mov, mkv
