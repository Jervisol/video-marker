from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
from loguru import logger

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['SECRET_KEY'] = 'your-secret-key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

socketio = SocketIO(app)

model_loaded = False
error_messages = []

try:
    model = YOLO('custom.pt')
    logger.info("Custom YOLO model detected")
    model_loaded = True
except Exception as e:
    error_messages.append(f"Error loading custom model: {e}")

if not model_loaded:
    try:
        model = YOLO('yolo11n.pt')
        logger.info("Successfully loaded YOLO11n model")
        model_loaded = True
    except Exception as e:
        error_messages.append(f"Error loading YOLO11n: {e}")

if not model_loaded:
    try:
        model = YOLO('yolov8n.pt')
        logger.info("Successfully loaded YOLOv8n model")
        model_loaded = True
    except Exception as e:
        error_messages.append(f"Error loading YOLOv8n: {e}")

if not model_loaded:
    logger.info("Failed to load any YOLO model:")
    for msg in error_messages:
        logger.info(f"  - {msg}")
    raise RuntimeError("model load failed")

CLASSES = model.names

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('index', video=filename))
    return redirect(request.url)

@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.json
    video_path = data.get('video_path')
    selected_classes = data.get('selected_classes', [])
    
    if not video_path:
        return jsonify({'error': 'No video path provided'})
    
    full_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
    
    if not os.path.exists(full_video_path):
        return jsonify({'error': 'Video file not found'})
    
    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_filename = f"output_{video_path}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        # 绘制检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = CLASSES[cls]
                
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
    
    return jsonify({'output_video': output_filename, 'frame_count': frame_count})

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    import base64
    from io import BytesIO
    
    data = request.json
    frame_data = data.get('frame_data')
    selected_classes = data.get('selected_classes', [])
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'})
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        results = model(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = CLASSES[cls]
                
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
        
        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)})


@socketio.on('detect_frame')
def handle_detect_frame(data):
    import base64
    from io import BytesIO
    
    frame_data = data.get('frame_data')
    selected_classes = data.get('selected_classes', [])
    
    if not frame_data:
        emit('detection_result', {'error': 'No frame data provided'})
        return
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        results = model(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = CLASSES[cls]
                
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
        
        # 通过WebSocket发送检测结果
        emit('detection_result', {'detections': detections})
    except Exception as e:
        emit('detection_result', {'error': str(e)})

@app.route('/detect_click', methods=['POST'])
def detect_click():
    import base64
    from io import BytesIO
    import math
    
    data = request.json
    frame_data = data.get('frame_data')
    click_x = data.get('click_x')
    click_y = data.get('click_y')
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'})
    
    if click_x is None or click_y is None:
        return jsonify({'error': 'No click coordinates provided'})
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        results = model(frame)
        
        closest_detection = None
        min_distance = float('inf')
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = CLASSES[cls]
                
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
                    return jsonify({'detected_class': class_name})
                
                # 更新最近的物体
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = class_name
        
        if closest_detection:
            return jsonify({'detected_class': closest_detection})
        else:
            return jsonify({'detected_class': None})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
