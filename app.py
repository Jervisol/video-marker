from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from config import Config
import os
import uuid
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pydantic import BaseModel

# 初始化FastAPI应用
app = FastAPI(title="Video Marker System", version="1.0.0")

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 线程池用于处理CPU密集型任务（模型推理）
executor = ThreadPoolExecutor(max_workers=4)

# 加载所有支持的模型
trained_models = {}

try:
    # 尝试从本地加载自定义模型
    custom_model = YOLO('custom.pt')
    trained_models['custom'] = custom_model
    logger.info("Custom YOLO model detected")
except Exception as e:
    logger.warning(f"Error loading custom model: {e}")

try:
    yolo11n_model = YOLO('yolo11n.pt')
    trained_models['yolo11n'] = yolo11n_model
    logger.info("Successfully loaded YOLO11n model")
except Exception as e:
    logger.warning(f"Error loading YOLO11n: {e}")

try:
    yolov8n_model = YOLO('yolov8n.pt')
    trained_models['yolov8n'] = yolov8n_model
    logger.info("Successfully loaded YOLOv8n model")
except Exception as e:
    logger.warning(f"Error loading YOLOv8n: {e}")

try:
    # 尝试加载EfficientDet模型（使用ultralytics支持的格式）
    efficientdet_model = YOLO('efficientdet-lite0.pt')
    trained_models['efficientdet'] = efficientdet_model
    logger.info("Successfully loaded EfficientDet model")
except Exception as e:
    logger.warning(f"Error loading EfficientDet: {e}")

# 过滤掉加载失败的模型（值为None的模型）
trained_models = {k: v for k, v in trained_models.items() if v is not None}

# 如果没有模型加载成功，抛出异常
if not trained_models:
    logger.error("Failed to load any model")
    raise RuntimeError("model load failed")

# 设置默认模型
default_model = Config.DEFAULT_MODEL if Config.DEFAULT_MODEL in trained_models else next(iter(trained_models.keys()))

# 类别名称（使用默认模型的类别）
CLASSES = trained_models[default_model].names

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

# ==================== Export Feature ====================

# Pydantic models for API requests/responses
class ExportStartRequest(BaseModel):
    video_filename: str
    selected_classes: List[str]
    selected_model: str
    output_filename: str
    output_fps: Optional[int] = None  # None表示使用原始帧率

class ExportStartResponse(BaseModel):
    task_id: str
    status: str
    total_frames: int

class ExportProgressResponse(BaseModel):
    task_id: str
    status: str
    processed_frames: int
    total_frames: int
    progress_percentage: float
    error_message: Optional[str] = None

class ExportCancelResponse(BaseModel):
    task_id: str
    status: str
    message: str

# Export Task data model
@dataclass
class ExportTask:
    task_id: str
    video_path: str
    output_path: str
    selected_classes: List[str]
    selected_model: str
    output_fps: Optional[int] = None  # None表示使用原始帧率
    status: str = "pending"
    processed_frames: int = 0
    total_frames: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    cancellation_flag: threading.Event = field(default_factory=threading.Event)
    
    def to_dict(self) -> dict:
        progress_pct = (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        return {
            "task_id": self.task_id,
            "status": self.status,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "progress_percentage": round(progress_pct, 2),
            "error_message": self.error_message
        }

# Export Task Manager
class ExportTaskManager:
    def __init__(self):
        self.tasks: Dict[str, ExportTask] = {}
        self.lock = threading.Lock()
    
    def create_task(self, video_path: str, output_path: str, 
                   selected_classes: List[str], selected_model: str,
                   output_fps: Optional[int] = None) -> ExportTask:
        task_id = str(uuid.uuid4())
        task = ExportTask(
            task_id=task_id,
            video_path=video_path,
            output_path=output_path,
            selected_classes=selected_classes,
            selected_model=selected_model,
            output_fps=output_fps
        )
        with self.lock:
            self.tasks[task_id] = task
        logger.info(f"Created export task {task_id}")
        return task
    
    def get_task(self, task_id: str) -> Optional[ExportTask]:
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_progress(self, task_id: str, processed_frames: int):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.processed_frames = processed_frames
    
    def mark_completed(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "completed"
                task.completed_at = datetime.now()
                logger.info(f"Task {task_id} completed")
    
    def mark_failed(self, task_id: str, error_message: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "failed"
                task.error_message = error_message
                task.completed_at = datetime.now()
                logger.error(f"Task {task_id} failed: {error_message}")
    
    def mark_cancelled(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "cancelled"
                task.completed_at = datetime.now()
                logger.info(f"Task {task_id} cancelled")
    
    def cancel_task(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == "processing":
                task.cancellation_flag.set()
                logger.info(f"Cancellation requested for task {task_id}")
                return True
            return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 1):
        """Remove tasks older than max_age_hours"""
        with self.lock:
            now = datetime.now()
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.completed_at:
                    age = (now - task.completed_at).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(task_id)
            for task_id in to_remove:
                del self.tasks[task_id]
                logger.info(f"Cleaned up old task {task_id}")

# Initialize export task manager
export_task_manager = ExportTaskManager()

# Export worker executor (separate from detection executor)
export_executor = ThreadPoolExecutor(max_workers=2)

# Export worker function
def export_worker(task: ExportTask):
    """
    Background worker that processes video frames and generates annotated output.
    Runs in a separate thread to avoid blocking the main application.
    """
    try:
        logger.info(f"Starting export worker for task {task.task_id}")
        
        # Update task status to processing
        with export_task_manager.lock:
            task.status = "processing"
        
        # Get the model
        if task.selected_model not in trained_models:
            raise ValueError(f"Model {task.selected_model} not found")
        
        current_model = trained_models[task.selected_model]
        
        # Open input video
        cap = cv2.VideoCapture(task.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {task.video_path}")
        
        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine output fps
        output_fps = task.output_fps if task.output_fps else input_fps
        
        # Calculate frame skip for fps reduction
        frame_skip = 1
        if task.output_fps and task.output_fps < input_fps:
            frame_skip = int(input_fps / task.output_fps)
        
        # Update total frames in task (adjusted for frame skip)
        adjusted_total_frames = total_frames // frame_skip if frame_skip > 1 else total_frames
        with export_task_manager.lock:
            task.total_frames = adjusted_total_frames
        
        logger.info(f"Video properties: {width}x{height}, input {input_fps}fps, output {output_fps}fps, {total_frames} frames")
        if frame_skip > 1:
            logger.info(f"Frame skip: {frame_skip}, adjusted frames: {adjusted_total_frames}")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(task.output_path, fourcc, output_fps, (width, height))
        
        if not out.isOpened():
            raise IOError(f"Cannot create output video file: {task.output_path}")
        
        frame_count = 0
        input_frame_count = 0
        
        # Process frames
        while cap.isOpened():
            # Check cancellation flag
            if task.cancellation_flag.is_set():
                logger.info(f"Task {task.task_id} cancellation detected")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            input_frame_count += 1
            
            # Skip frames if fps reduction is enabled
            if frame_skip > 1 and (input_frame_count - 1) % frame_skip != 0:
                continue
            
            # Run detection
            results = current_model(frame)
            
            # Draw bounding boxes
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = current_model.names[cls]
                    
                    # Only process selected classes
                    if class_name in task.selected_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Write annotated frame
            out.write(frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 10 == 0:  # Update every 10 frames to reduce lock contention
                export_task_manager.update_progress(task.task_id, frame_count)
        
        # Final progress update
        export_task_manager.update_progress(task.task_id, frame_count)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Mark as completed or cancelled
        if task.cancellation_flag.is_set():
            # Delete incomplete output file
            if os.path.exists(task.output_path):
                os.remove(task.output_path)
                logger.info(f"Deleted incomplete output file: {task.output_path}")
            export_task_manager.mark_cancelled(task.task_id)
        else:
            export_task_manager.mark_completed(task.task_id)
            logger.info(f"Export completed: {task.output_path}")
            
    except Exception as e:
        logger.exception(f"Export worker error for task {task.task_id}")
        export_task_manager.mark_failed(task.task_id, str(e))
        
        # Clean up partial output file
        if os.path.exists(task.output_path):
            try:
                os.remove(task.output_path)
                logger.info(f"Deleted partial output file: {task.output_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to delete partial file: {cleanup_error}")

# 模型推理函数（CPU密集型，在单独线程中执行）
def detect_with_model(frame: np.ndarray, selected_model: str, selected_classes: list):
    if selected_model not in trained_models:
        raise ValueError(f"Model {selected_model} not found")
    
    current_model = trained_models[selected_model]
    results = current_model(frame)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # 使用当前模型的类别名称
            class_name = current_model.names[cls]
            
            # 只处理选中的类别
            if selected_classes and class_name not in selected_classes:
                continue
            
            # 获取边界框坐标和置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # 添加到检测结果列表
            detections.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'class_name': class_name,
                'confidence': conf
            })
    
    return detections

# 页面路由
@app.get('/')
def index(request: Request):
    # 获取查询参数中的video
    video = request.query_params.get('video')
    # 只传递模型名称列表给模板，避免模板渲染时的序列化问题
    model_names = list(trained_models.keys())
    return templates.TemplateResponse('index.html', {
        'request': request,
        'classes': CLASSES,
        'trained_models': model_names,
        'default_model': default_model,
        'video': video
    })

# WebSocket路由
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            frame_data = data.get('frame_data')
            selected_classes = data.get('selected_classes', [])
            selected_model = data.get('selected_model', default_model)
            
            if not frame_data:
                await manager.send_personal_message({'error': 'No frame data provided'}, websocket)
                continue
            
            try:
                # 将base64编码的帧数据转换为OpenCV格式的图像
                img_data = base64.b64decode(frame_data)
                img = Image.open(BytesIO(img_data))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # 在单独线程中执行模型推理
                detections = await asyncio.get_event_loop().run_in_executor(
                    executor, detect_with_model, frame, selected_model, selected_classes
                )
                
                # 发送检测结果
                await manager.send_personal_message({'detections': detections}, websocket)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await manager.send_personal_message({'error': str(e)}, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# HTTP API路由：单帧检测
@app.post('/detect_frame')
async def detect_frame_api(request: Request):
    data = await request.json()
    frame_data = data.get('frame_data')
    selected_classes = data.get('selected_classes', [])
    selected_model = data.get('selected_model', default_model)
    
    if not frame_data:
        raise HTTPException(status_code=400, detail="No frame data provided")
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 在单独线程中执行模型推理
        detections = await asyncio.get_event_loop().run_in_executor(
            executor, detect_with_model, frame, selected_model, selected_classes
        )
        
        return {'detections': detections}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# HTTP API路由：点击检测
@app.post('/detect_click')
async def detect_click_api(request: Request):
    data = await request.json()
    frame_data = data.get('frame_data')
    click_x = data.get('click_x')
    click_y = data.get('click_y')
    selected_model = data.get('selected_model', default_model)
    
    if not frame_data:
        raise HTTPException(status_code=400, detail="No frame data provided")
    
    if click_x is None or click_y is None:
        raise HTTPException(status_code=400, detail="No click coordinates provided")
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 获取选择的模型
        if selected_model not in trained_models:
            raise ValueError(f"Model {selected_model} not found")
        
        current_model = trained_models[selected_model]
        results = current_model(frame)
        
        closest_detection = None
        min_distance = float('inf')
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                # 使用当前模型的类别名称
                class_name = current_model.names[cls]
                
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 计算物体中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 计算点击坐标到物体中心点的距离
                distance = math.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)
                
                # 检查点击是否在物体边界框内
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    # 如果点击在边界框内，直接返回该物体
                    return {'detected_class': class_name}
                
                # 更新最近的物体
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = class_name
        
        if closest_detection:
            return {'detected_class': closest_detection}
        else:
            return {'detected_class': None}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# HTTP API路由：视频检测（保留原功能，实际项目中建议使用更高效的方式）
@app.post('/detect')
async def detect_objects_api(request: Request):
    data = await request.json()
    video_path = data.get('video_path')
    selected_classes = data.get('selected_classes', [])
    selected_model = data.get('selected_model', default_model)
    
    if not video_path:
        raise HTTPException(status_code=400, detail="No video path provided")
    
    full_video_path = f"{Config.UPLOAD_FOLDER}/{video_path}"
    
    if not os.path.exists(full_video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # 获取选择的模型
    if selected_model not in trained_models:
        raise HTTPException(status_code=400, detail=f"Model {selected_model} not found")
    
    current_model = trained_models[selected_model]
    
    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_filename = f"output_{video_path}"
    output_path = f"{Config.UPLOAD_FOLDER}/{output_filename}"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = current_model(frame)
        
        # 绘制检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                # 使用当前模型的类别名称
                class_name = current_model.names[cls]
                
                # 只处理选中的类别
                if selected_classes and class_name not in selected_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加类别标签和置信度
                conf = float(box.conf[0])
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return {'output_video': output_filename, 'frame_count': frame_count}

# ==================== Video Metadata API ====================

class VideoMetadataRequest(BaseModel):
    video_filename: str

class VideoMetadataResponse(BaseModel):
    fps: float
    width: int
    height: int
    total_frames: int
    duration: float

@app.post('/video/metadata', response_model=VideoMetadataResponse)
async def get_video_metadata(request: VideoMetadataRequest):
    """
    Get video metadata including fps, resolution, duration, and frame count.
    """
    try:
        video_path = f"{Config.UPLOAD_FOLDER}/{request.video_filename}"
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_filename}")
        
        # Open video to get metadata
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoMetadataResponse(
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting video metadata")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Export API Endpoints ====================

@app.post('/export/start', response_model=ExportStartResponse)
async def start_export(request: ExportStartRequest):
    """
    Start a new export task.
    Creates a background task to process the video and generate annotated output.
    """
    try:
        # Validate video file exists
        video_path = f"{Config.UPLOAD_FOLDER}/{request.video_filename}"
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_filename}")
        
        # Validate selected classes
        if not request.selected_classes:
            raise HTTPException(status_code=400, detail="At least one class must be selected")
        
        # Validate model
        if request.selected_model not in trained_models:
            raise HTTPException(status_code=400, detail=f"Model not found: {request.selected_model}")
        
        # Create output path
        output_path = f"{Config.UPLOAD_FOLDER}/{request.output_filename}"
        
        # Check if output file already exists
        if os.path.exists(output_path):
            # Add timestamp to make it unique
            base, ext = os.path.splitext(request.output_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base}_{timestamp}{ext}"
            output_path = f"{Config.UPLOAD_FOLDER}/{output_filename}"
        
        # Create export task
        task = export_task_manager.create_task(
            video_path=video_path,
            output_path=output_path,
            selected_classes=request.selected_classes,
            selected_model=request.selected_model,
            output_fps=request.output_fps
        )
        
        # Get total frames for response
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        task.total_frames = total_frames
        
        # Submit task to executor
        export_executor.submit(export_worker, task)
        
        logger.info(f"Export task {task.task_id} started")
        
        return ExportStartResponse(
            task_id=task.task_id,
            status="started",
            total_frames=total_frames
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error starting export")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/export/progress/{task_id}', response_model=ExportProgressResponse)
async def get_export_progress(task_id: str):
    """
    Get the progress of an export task.
    Returns current status, processed frames, and progress percentage.
    """
    try:
        task = export_task_manager.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        return ExportProgressResponse(**task.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting progress for task {task_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/export/cancel/{task_id}', response_model=ExportCancelResponse)
async def cancel_export(task_id: str):
    """
    Cancel an ongoing export task.
    Sets the cancellation flag to stop the worker and clean up resources.
    """
    try:
        task = export_task_manager.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        if task.status not in ["pending", "processing"]:
            return ExportCancelResponse(
                task_id=task_id,
                status=task.status,
                message=f"Task cannot be cancelled (current status: {task.status})"
            )
        
        # Set cancellation flag
        success = export_task_manager.cancel_task(task_id)
        
        if success:
            return ExportCancelResponse(
                task_id=task_id,
                status="cancelling",
                message="Cancellation requested"
            )
        else:
            return ExportCancelResponse(
                task_id=task_id,
                status=task.status,
                message="Task could not be cancelled"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error cancelling task {task_id}")
        raise HTTPException(status_code=500, detail=str(e))

# 上传视频路由
@app.post('/upload')
async def upload_video(request: Request):
    form = await request.form()
    # 只传递模型名称列表给模板，避免模板渲染时的序列化问题
    model_names = list(trained_models.keys())
    if 'video' not in form:
        return templates.TemplateResponse('index.html', {
            'request': request,
            'classes': CLASSES,
            'trained_models': model_names,
            'default_model': default_model
        })
    
    file = form['video']
    if file.filename == '':
        return templates.TemplateResponse('index.html', {
            'request': request,
            'classes': CLASSES,
            'trained_models': model_names,
            'default_model': default_model
        })
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = f"{Config.UPLOAD_FOLDER}/{filename}"
        
        # 保存文件
        with open(filepath, "wb") as f:
            f.write(await file.read())
        
        return templates.TemplateResponse('index.html', {
            'request': request,
            'classes': CLASSES,
            'trained_models': model_names,
            'default_model': default_model,
            'video': filename
        })
    
    return templates.TemplateResponse('index.html', {
        'request': request,
        'classes': CLASSES,
        'trained_models': model_names,
        'default_model': default_model
    })

if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app, host='0.0.0.0', port=5000, reload=True)
    uvicorn.run("app:app", host="0.0.0.0", port=5055, reload=True)