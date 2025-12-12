import os

class Config:
    # 上传文件配置
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    
    # 应用配置
    SECRET_KEY = 'your-secret-key'  # 用于WebSocket加密
    
    # 模型配置
    DEFAULT_MODEL = 'yolo11n'
    
    # 确保上传目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
