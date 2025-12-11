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
app.config['SECRET_KEY'] = 'your-secret-key'  # 用于WebSocket加密

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化SocketIO
socketio = SocketIO(app)

# 加载所有支持的模型
try:
    # 尝试从本地加载自定义模型
    custom_model = YOLO('custom.pt')
    logger.info("Custom YOLO model detected")
except Exception as e:
    custom_model = None
    logger.warning(f"Error loading custom model: {e}")

try:
    yolo11n_model = YOLO('yolo11n.pt')
    logger.info("Successfully loaded YOLO11n model")
except Exception as e:
    yolo11n_model = None
    logger.warning(f"Error loading YOLO11n: {e}")

try:
    yolov8n_model = YOLO('yolov8n.pt')
    logger.info("Successfully loaded YOLOv8n model")
except Exception as e:
    yolov8n_model = None
    logger.warning(f"Error loading YOLOv8n: {e}")

try:
    # 尝试加载EfficientDet模型（使用ultralytics支持的格式）
    efficientdet_model = YOLO('efficientdet-lite0.pt')
    logger.info("Successfully loaded EfficientDet model")
except Exception as e:
    efficientdet_model = None
    logger.warning(f"Error loading EfficientDet: {e}")

# 模型映射
trained_models = {
    'yolo11n': yolo11n_model,
    'yolov8n': yolov8n_model,
    'custom': custom_model,
    'efficientdet': efficientdet_model
}

# 过滤掉加载失败的模型
trained_models = {k: v for k, v in trained_models.items() if v is not None}

# 如果没有模型加载成功，抛出异常
if not trained_models:
    logger.error("Failed to load any model")
    raise RuntimeError("model load failed")

# 设置默认模型
default_model = next(iter(trained_models.keys()))

# 类别名称（使用默认模型的类别）
CLASSES = trained_models[default_model].names

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES, trained_models=trained_models, default_model=default_model)

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
    selected_model = data.get('selected_model', default_model)
    
    if not video_path:
        return jsonify({'error': 'No video path provided'})
    
    full_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
    
    if not os.path.exists(full_video_path):
        return jsonify({'error': 'Video file not found'})
    
    # 获取选择的模型
    current_model = trained_models[selected_model]
    
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
    
    return jsonify({'output_video': output_filename, 'frame_count': frame_count})

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    import base64
    from io import BytesIO
    
    data = request.json
    frame_data = data.get('frame_data')
    selected_classes = data.get('selected_classes', [])
    selected_model = data.get('selected_model', default_model)
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'})
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 获取选择的模型
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
        
        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)})


@socketio.on('detect_frame')
def handle_detect_frame(data):
    import base64
    from io import BytesIO
    
    frame_data = data.get('frame_data')
    selected_classes = data.get('selected_classes', [])
    selected_model = data.get('selected_model', default_model)
    
    if not frame_data:
        emit('detection_result', {'error': 'No frame data provided'})
        return
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 获取选择的模型
        current_model = trained_models[selected_model]
        
        # 使用选择的模型进行检测
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
    selected_model = data.get('selected_model', default_model)
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'})
    
    if click_x is None or click_y is None:
        return jsonify({'error': 'No click coordinates provided'})
    
    try:
        # 将base64编码的帧数据转换为OpenCV格式的图像
        img_data = base64.b64decode(frame_data)
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 获取选择的模型
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
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
