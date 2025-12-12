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
    # 只传递模型名称列表给模板，避免模板渲染时的序列化问题
    model_names = list(trained_models.keys())
    return templates.TemplateResponse('index.html', {
        'request': request,
        'classes': CLASSES,
        'trained_models': model_names,
        'default_model': default_model
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
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)