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


class TrainingConfig:
    # 训练模块目录配置
    ANNOTATIONS_FOLDER = 'static/training/annotations'
    DATASETS_FOLDER = 'static/training/datasets'
    MODELS_FOLDER = 'static/training/models'
    
    # 限制配置
    MAX_ANNOTATION_SIZE_MB = 10
    MAX_CONCURRENT_TRAINING = 2
    
    # LoRA默认参数
    DEFAULT_LORA_RANK = 8
    DEFAULT_LORA_ALPHA = 16
    DEFAULT_LORA_DROPOUT = 0.1
    
    # 训练默认参数
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_IMAGE_SIZE = 640
    
    # 确保训练目录存在
    os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
    os.makedirs(DATASETS_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
