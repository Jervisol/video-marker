# Project Structure

## Root Files

- `app.py` - Main FastAPI application with all routes and WebSocket handlers
- `config.py` - Configuration settings (upload folder, allowed extensions, default model)
- `requirements.txt` - Python dependencies
- `*.pt` - YOLO model weight files (yolo11n.pt, yolov8n.pt)

## Directories

### `/templates`
HTML templates using Jinja2 templating
- `index.html` - Main single-page application with video player, controls, and detection UI

### `/static/uploads`
Storage for uploaded videos and exported annotated videos
- Original uploaded videos
- Generated output videos with `output_` prefix
- Generated annotated videos with `annotated_` prefix

### `/docs`
Documentation for features and optimizations
- Export feature documentation
- Video export optimization notes

### `/bak`
Backup files and historical documentation
- Previous versions of code
- Feature implementation notes
- Timeline feature documentation

### `/.kiro`
Kiro IDE configuration
- `/specs` - Feature specifications (realtime-video-detection, video-export)
- `/steering` - AI assistant guidance documents (this directory)

## Code Organization

### app.py Structure

1. **Imports & Initialization** - FastAPI app, model loading, static files
2. **Model Management** - Load YOLO models, handle failures gracefully
3. **WebSocket Manager** - ConnectionManager class for real-time detection
4. **Export Feature** - Pydantic models, ExportTask dataclass, ExportTaskManager
5. **Detection Functions** - `detect_with_model()` for frame inference
6. **Routes**:
   - `GET /` - Main page
   - `WebSocket /ws` - Real-time frame detection
   - `POST /detect_frame` - Single frame detection API
   - `POST /detect_click` - Click-based object detection
   - `POST /detect` - Full video processing (legacy)
   - `POST /video/metadata` - Get video properties
   - `POST /export/start` - Start background export task
   - `GET /export/progress/{task_id}` - Check export progress
   - `POST /export/cancel/{task_id}` - Cancel export task
   - `POST /upload` - Video file upload

## Naming Conventions

- **Routes**: Use snake_case for endpoint names
- **Functions**: snake_case for all Python functions
- **Classes**: PascalCase for classes (ExportTask, ConnectionManager)
- **Variables**: snake_case for variables
- **Constants**: UPPER_SNAKE_CASE (CLASSES, ALLOWED_EXTENSIONS)
- **Files**: snake_case for Python files, lowercase for HTML
- **Output Videos**: Prefix with `output_` or `annotated_` followed by original filename
