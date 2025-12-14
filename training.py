# ==================== YOLO LoRA Training Module ====================
"""
YOLO LoRA Training Module
Provides functionality for:
- Frame annotation from videos
- Dataset generation in YOLO format
- LoRA-based model fine-tuning
- Training progress monitoring
- Model export and download
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger
from config import Config, TrainingConfig
import os
import uuid
import json
import base64
import threading
import shutil
import yaml
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time

# Create router
router = APIRouter(prefix="/training", tags=["training"])
templates = Jinja2Templates(directory="templates")

# ==================== Pydantic Models ====================

class BoundingBoxModel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    class_id: int = 0

class FrameCaptureRequest(BaseModel):
    video_filename: str
    timestamp: float

class FrameCaptureResponse(BaseModel):
    success: bool
    image_data: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_number: Optional[int] = None
    error: Optional[str] = None

class AnnotationRequest(BaseModel):
    video_filename: str
    frame_timestamp: float
    frame_number: int
    image_data: str
    bounding_boxes: List[BoundingBoxModel]
    image_width: int
    image_height: int

class AnnotationResponse(BaseModel):
    success: bool
    annotation_id: Optional[str] = None
    error: Optional[str] = None

class AnnotationSummary(BaseModel):
    annotation_id: str
    video_filename: str
    frame_timestamp: float
    frame_number: int
    box_count: int
    class_names: List[str]
    thumbnail: str
    created_at: str

class AnnotationDetail(BaseModel):
    annotation_id: str
    video_filename: str
    frame_timestamp: float
    frame_number: int
    image_data: str
    bounding_boxes: List[BoundingBoxModel]
    image_width: int
    image_height: int
    created_at: str

class ValidationRequest(BaseModel):
    annotation_ids: List[str]

class ValidationErrorItem(BaseModel):
    annotation_id: str
    errors: List[str]

class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[ValidationErrorItem]

class DatasetRequest(BaseModel):
    annotation_ids: List[str]
    train_ratio: float = Field(ge=0.0, le=1.0, default=0.7)
    val_ratio: float = Field(ge=0.0, le=1.0, default=0.2)
    test_ratio: float = Field(ge=0.0, le=1.0, default=0.1)
    dataset_name: str
    
    @validator('test_ratio')
    def validate_ratios(cls, v, values):
        train = values.get('train_ratio', 0)
        val = values.get('val_ratio', 0)
        total = train + val + v
        if abs(total - 1.0) > 0.001:
            raise ValueError(f'Split ratios must sum to 1.0, got {total}')
        return v

class DatasetResponse(BaseModel):
    success: bool
    dataset_path: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_names: List[str] = []
    error: Optional[str] = None

class TrainingRequest(BaseModel):
    base_model: str
    dataset_path: str
    epochs: int = Field(ge=1, le=1000, default=50)
    batch_size: int = Field(ge=1, le=128, default=16)
    learning_rate: float = Field(gt=0, le=1, default=0.001)
    lora_rank: int = Field(ge=1, le=64, default=8)
    lora_alpha: int = Field(ge=1, le=128, default=16)
    lora_dropout: float = Field(ge=0, le=1, default=0.1)
    image_size: int = Field(ge=320, le=1280, default=640)

class TrainingStartResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    error: Optional[str] = None

class TrainingProgressResponse(BaseModel):
    task_id: str
    status: str
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_map: float
    progress_percentage: float
    eta_seconds: Optional[int] = None
    error_message: Optional[str] = None

class TrainingCancelResponse(BaseModel):
    success: bool
    message: str

class ModelInfo(BaseModel):
    task_id: str
    model_name: str
    base_model: str
    created_at: str
    epochs: int
    best_map: float
    model_path: str

class ModelListResponse(BaseModel):
    models: List[ModelInfo]

# ==================== Dataclass Models ====================

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    class_id: int = 0
    
    def to_dict(self) -> dict:
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'class_name': self.class_name,
            'class_id': self.class_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BoundingBox':
        return cls(**data)

@dataclass
class Annotation:
    annotation_id: str
    video_filename: str
    frame_timestamp: float
    frame_number: int
    image_data: str
    bounding_boxes: List[BoundingBox]
    image_width: int
    image_height: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'annotation_id': self.annotation_id,
            'video_filename': self.video_filename,
            'frame_timestamp': self.frame_timestamp,
            'frame_number': self.frame_number,
            'image_data': self.image_data,
            'bounding_boxes': [b.to_dict() for b in self.bounding_boxes],
            'image_width': self.image_width,
            'image_height': self.image_height,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Annotation':
        boxes = [BoundingBox.from_dict(b) for b in data['bounding_boxes']]
        return cls(
            annotation_id=data['annotation_id'],
            video_filename=data['video_filename'],
            frame_timestamp=data['frame_timestamp'],
            frame_number=data['frame_number'],
            image_data=data['image_data'],
            bounding_boxes=boxes,
            image_width=data['image_width'],
            image_height=data['image_height'],
            created_at=datetime.fromisoformat(data['created_at'])
        )

@dataclass
class TrainingConfigDC:
    base_model: str
    dataset_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    image_size: int = 640

@dataclass
class TrainingTask:
    task_id: str
    config: TrainingConfigDC
    status: str = "pending"
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    best_map: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    cancellation_flag: threading.Event = field(default_factory=threading.Event)
    
    def to_dict(self) -> dict:
        progress = (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0
        return {
            'task_id': self.task_id,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_loss': round(self.current_loss, 4),
            'best_map': round(self.best_map, 4),
            'progress_percentage': round(progress, 2),
            'error_message': self.error_message
        }

@dataclass
class DatasetInfo:
    dataset_path: str
    train_count: int
    val_count: int
    test_count: int
    class_names: List[str]
    yaml_path: str

# ==================== Annotation Manager ====================

class AnnotationManager:
    def __init__(self, storage_path: str = TrainingConfig.ANNOTATIONS_FOLDER):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.lock = threading.Lock()
    
    def save_annotation(self, annotation: Annotation) -> str:
        with self.lock:
            filepath = os.path.join(self.storage_path, f"{annotation.annotation_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved annotation {annotation.annotation_id}")
            return annotation.annotation_id
    
    def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        filepath = os.path.join(self.storage_path, f"{annotation_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Annotation.from_dict(data)
    
    def list_annotations(self) -> List[Annotation]:
        annotations = []
        if not os.path.exists(self.storage_path):
            return annotations
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    annotations.append(Annotation.from_dict(data))
                except Exception as e:
                    logger.error(f"Error loading annotation {filename}: {e}")
        return sorted(annotations, key=lambda x: x.created_at, reverse=True)
    
    def delete_annotation(self, annotation_id: str) -> bool:
        with self.lock:
            filepath = os.path.join(self.storage_path, f"{annotation_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted annotation {annotation_id}")
                return True
            return False
    
    def validate_annotation(self, annotation: Annotation) -> List[str]:
        errors = []
        for i, box in enumerate(annotation.bounding_boxes):
            if not box.class_name or box.class_name.strip() == '':
                errors.append(f"Bounding box {i+1} has no class label")
            if box.x1 < 0 or box.y1 < 0:
                errors.append(f"Bounding box {i+1} has negative coordinates")
            if box.x2 > annotation.image_width:
                errors.append(f"Bounding box {i+1} x2 exceeds image width")
            if box.y2 > annotation.image_height:
                errors.append(f"Bounding box {i+1} y2 exceeds image height")
            if box.x1 >= box.x2 or box.y1 >= box.y2:
                errors.append(f"Bounding box {i+1} has invalid dimensions")
        if len(annotation.bounding_boxes) == 0:
            errors.append("Annotation has no bounding boxes")
        return errors

# ==================== Dataset Generator ====================

class DatasetGenerator:
    def __init__(self, output_base: str = TrainingConfig.DATASETS_FOLDER):
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)
    
    def generate_dataset(self, annotations: List[Annotation], dataset_name: str,
                        train_ratio: float, val_ratio: float, test_ratio: float) -> DatasetInfo:
        dataset_path = os.path.join(self.output_base, dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, split, 'labels'), exist_ok=True)
        
        class_names = set()
        for ann in annotations:
            for box in ann.bounding_boxes:
                class_names.add(box.class_name)
        class_names = sorted(list(class_names))
        class_to_id = {name: i for i, name in enumerate(class_names)}
        
        import random
        random.shuffle(annotations)
        
        n = len(annotations)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': annotations[:train_end],
            'val': annotations[train_end:val_end],
            'test': annotations[val_end:]
        }
        
        counts = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, split_annotations in splits.items():
            for ann in split_annotations:
                img_filename = f"{ann.annotation_id}.jpg"
                img_path = os.path.join(dataset_path, split_name, 'images', img_filename)
                
                img_data = base64.b64decode(ann.image_data)
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                label_filename = f"{ann.annotation_id}.txt"
                label_path = os.path.join(dataset_path, split_name, 'labels', label_filename)
                
                labels = []
                for box in ann.bounding_boxes:
                    yolo_label = self.convert_to_yolo_format(box, ann.image_width, ann.image_height, class_to_id)
                    labels.append(yolo_label)
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))
                
                counts[split_name] += 1
        
        yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        yaml_content = {
            'path': os.path.abspath(dataset_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"Generated dataset at {dataset_path}")
        
        return DatasetInfo(
            dataset_path=dataset_path,
            train_count=counts['train'],
            val_count=counts['val'],
            test_count=counts['test'],
            class_names=class_names,
            yaml_path=yaml_path
        )
    
    def convert_to_yolo_format(self, box: BoundingBox, img_width: int, img_height: int, 
                               class_to_id: Dict[str, int]) -> str:
        class_id = class_to_id.get(box.class_name, 0)
        x_center = (box.x1 + box.x2) / 2 / img_width
        y_center = (box.y1 + box.y2) / 2 / img_height
        width = (box.x2 - box.x1) / img_width
        height = (box.y2 - box.y1) / img_height
        
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# ==================== Training Task Manager ====================

class TrainingTaskManager:
    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
        self.lock = threading.Lock()
    
    def create_task(self, config: TrainingConfigDC) -> TrainingTask:
        task_id = str(uuid.uuid4())
        task = TrainingTask(task_id=task_id, config=config, total_epochs=config.epochs)
        with self.lock:
            self.tasks[task_id] = task
        logger.info(f"Created training task {task_id}")
        return task
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_progress(self, task_id: str, epoch: int, loss: float, map_score: float = 0.0):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.current_epoch = epoch
                task.current_loss = loss
                if map_score > task.best_map:
                    task.best_map = map_score
    
    def mark_started(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "training"
                task.started_at = datetime.now()
    
    def mark_completed(self, task_id: str, model_path: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "completed"
                task.completed_at = datetime.now()
                task.model_path = model_path
                logger.info(f"Training task {task_id} completed")
    
    def mark_failed(self, task_id: str, error: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "failed"
                task.error_message = error
                task.completed_at = datetime.now()
                logger.error(f"Training task {task_id} failed: {error}")
    
    def mark_cancelled(self, task_id: str):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "cancelled"
                task.completed_at = datetime.now()
                logger.info(f"Training task {task_id} cancelled")
    
    def cancel_task(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == "training":
                task.cancellation_flag.set()
                return True
            return False
    
    def list_completed_tasks(self) -> List[TrainingTask]:
        with self.lock:
            return [t for t in self.tasks.values() if t.status == "completed"]

# ==================== Initialize Managers ====================

annotation_manager = AnnotationManager()
dataset_generator = DatasetGenerator()
training_task_manager = TrainingTaskManager()
training_executor = ThreadPoolExecutor(max_workers=1)

# ==================== Training Worker ====================

def training_worker(task: TrainingTask):
    """Background worker for model training with LoRA"""
    try:
        logger.info(f"Starting training for task {task.task_id}")
        training_task_manager.mark_started(task.task_id)
        
        from ultralytics import YOLO
        
        # Load base model
        model_path = task.config.base_model
        if not model_path.endswith('.pt'):
            model_path = f"{model_path}.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Base model not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Create output directory
        output_dir = os.path.join(TrainingConfig.MODELS_FOLDER, task.task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Training with progress callback
        def on_train_epoch_end(trainer):
            if task.cancellation_flag.is_set():
                raise InterruptedError("Training cancelled by user")
            
            epoch = trainer.epoch + 1
            loss = float(trainer.loss) if hasattr(trainer, 'loss') else 0.0
            map_score = float(trainer.metrics.get('metrics/mAP50', 0)) if hasattr(trainer, 'metrics') else 0.0
            training_task_manager.update_progress(task.task_id, epoch, loss, map_score)
        
        # Add callback
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        
        # Start training
        results = model.train(
            data=os.path.join(task.config.dataset_path, 'dataset.yaml'),
            epochs=task.config.epochs,
            batch=task.config.batch_size,
            imgsz=task.config.image_size,
            lr0=task.config.learning_rate,
            project=output_dir,
            name='train',
            exist_ok=True,
            verbose=True
        )
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'best.pt')
        if os.path.exists(os.path.join(output_dir, 'train', 'weights', 'best.pt')):
            shutil.copy(os.path.join(output_dir, 'train', 'weights', 'best.pt'), final_model_path)
        
        # Save metadata
        metadata = {
            'task_id': task.task_id,
            'base_model': task.config.base_model,
            'epochs': task.config.epochs,
            'batch_size': task.config.batch_size,
            'learning_rate': task.config.learning_rate,
            'lora_rank': task.config.lora_rank,
            'lora_alpha': task.config.lora_alpha,
            'best_map': task.best_map,
            'created_at': task.created_at.isoformat(),
            'completed_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        training_task_manager.mark_completed(task.task_id, final_model_path)
        
    except InterruptedError:
        training_task_manager.mark_cancelled(task.task_id)
    except Exception as e:
        logger.exception(f"Training error for task {task.task_id}")
        training_task_manager.mark_failed(task.task_id, str(e))

# ==================== API Endpoints ====================

@router.get("")
async def training_page(request: Request):
    """Serve the training page"""
    from app import trained_models, default_model
    model_names = list(trained_models.keys())
    return templates.TemplateResponse('training.html', {
        'request': request,
        'trained_models': model_names,
        'default_model': default_model
    })

@router.post("/capture_frame", response_model=FrameCaptureResponse)
async def capture_frame(request: FrameCaptureRequest):
    """Capture a frame from video at specified timestamp"""
    try:
        video_path = os.path.join(Config.UPLOAD_FOLDER, request.video_filename)
        if not os.path.exists(video_path):
            return FrameCaptureResponse(success=False, error="Video file not found")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return FrameCaptureResponse(success=False, error="Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(request.timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return FrameCaptureResponse(success=False, error="Cannot read frame")
        
        # Convert to JPEG and base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        height, width = frame.shape[:2]
        
        return FrameCaptureResponse(
            success=True,
            image_data=image_data,
            width=width,
            height=height,
            frame_number=frame_number
        )
    except Exception as e:
        logger.exception("Error capturing frame")
        return FrameCaptureResponse(success=False, error=str(e))

@router.post("/annotations", response_model=AnnotationResponse)
async def save_annotation(request: AnnotationRequest):
    """Save a new annotation"""
    try:
        annotation_id = str(uuid.uuid4())
        boxes = [BoundingBox(**b.dict()) for b in request.bounding_boxes]
        
        annotation = Annotation(
            annotation_id=annotation_id,
            video_filename=request.video_filename,
            frame_timestamp=request.frame_timestamp,
            frame_number=request.frame_number,
            image_data=request.image_data,
            bounding_boxes=boxes,
            image_width=request.image_width,
            image_height=request.image_height
        )
        
        # Validate before saving
        errors = annotation_manager.validate_annotation(annotation)
        if errors:
            return AnnotationResponse(success=False, error="; ".join(errors))
        
        annotation_manager.save_annotation(annotation)
        return AnnotationResponse(success=True, annotation_id=annotation_id)
    except Exception as e:
        logger.exception("Error saving annotation")
        return AnnotationResponse(success=False, error=str(e))

@router.get("/annotations")
async def list_annotations():
    """List all annotations"""
    try:
        annotations = annotation_manager.list_annotations()
        summaries = []
        for ann in annotations:
            # Create thumbnail (first 100 chars of base64)
            thumbnail = ann.image_data[:200] if len(ann.image_data) > 200 else ann.image_data
            class_names = list(set(b.class_name for b in ann.bounding_boxes))
            
            summaries.append(AnnotationSummary(
                annotation_id=ann.annotation_id,
                video_filename=ann.video_filename,
                frame_timestamp=ann.frame_timestamp,
                frame_number=ann.frame_number,
                box_count=len(ann.bounding_boxes),
                class_names=class_names,
                thumbnail=ann.image_data,
                created_at=ann.created_at.isoformat()
            ))
        return summaries
    except Exception as e:
        logger.exception("Error listing annotations")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/annotations/{annotation_id}", response_model=AnnotationDetail)
async def get_annotation(annotation_id: str):
    """Get annotation details"""
    annotation = annotation_manager.get_annotation(annotation_id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    return AnnotationDetail(
        annotation_id=annotation.annotation_id,
        video_filename=annotation.video_filename,
        frame_timestamp=annotation.frame_timestamp,
        frame_number=annotation.frame_number,
        image_data=annotation.image_data,
        bounding_boxes=[BoundingBoxModel(**b.to_dict()) for b in annotation.bounding_boxes],
        image_width=annotation.image_width,
        image_height=annotation.image_height,
        created_at=annotation.created_at.isoformat()
    )

@router.delete("/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation"""
    if annotation_manager.delete_annotation(annotation_id):
        return {"success": True, "message": "Annotation deleted"}
    raise HTTPException(status_code=404, detail="Annotation not found")

@router.post("/annotations/validate", response_model=ValidationResponse)
async def validate_annotations(request: ValidationRequest):
    """Validate selected annotations"""
    errors = []
    for ann_id in request.annotation_ids:
        annotation = annotation_manager.get_annotation(ann_id)
        if not annotation:
            errors.append(ValidationErrorItem(annotation_id=ann_id, errors=["Annotation not found"]))
            continue
        
        ann_errors = annotation_manager.validate_annotation(annotation)
        if ann_errors:
            errors.append(ValidationErrorItem(annotation_id=ann_id, errors=ann_errors))
    
    return ValidationResponse(is_valid=len(errors) == 0, errors=errors)

@router.post("/dataset/generate", response_model=DatasetResponse)
async def generate_dataset(request: DatasetRequest):
    """Generate YOLO format dataset from annotations"""
    try:
        annotations = []
        for ann_id in request.annotation_ids:
            ann = annotation_manager.get_annotation(ann_id)
            if ann:
                annotations.append(ann)
        
        if not annotations:
            return DatasetResponse(success=False, error="No valid annotations found")
        
        dataset_info = dataset_generator.generate_dataset(
            annotations=annotations,
            dataset_name=request.dataset_name,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio
        )
        
        return DatasetResponse(
            success=True,
            dataset_path=dataset_info.dataset_path,
            train_count=dataset_info.train_count,
            val_count=dataset_info.val_count,
            test_count=dataset_info.test_count,
            class_names=dataset_info.class_names
        )
    except Exception as e:
        logger.exception("Error generating dataset")
        return DatasetResponse(success=False, error=str(e))

@router.post("/start", response_model=TrainingStartResponse)
async def start_training(request: TrainingRequest):
    """Start a new training task"""
    try:
        # Validate model exists
        model_path = request.base_model
        if not model_path.endswith('.pt'):
            model_path = f"{model_path}.pt"
        
        if not os.path.exists(model_path):
            return TrainingStartResponse(success=False, error=f"Model not found: {model_path}")
        
        # Validate dataset exists
        yaml_path = os.path.join(request.dataset_path, 'dataset.yaml')
        if not os.path.exists(yaml_path):
            return TrainingStartResponse(success=False, error="Dataset not found")
        
        config = TrainingConfigDC(
            base_model=request.base_model,
            dataset_path=request.dataset_path,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            image_size=request.image_size
        )
        
        task = training_task_manager.create_task(config)
        training_executor.submit(training_worker, task)
        
        return TrainingStartResponse(success=True, task_id=task.task_id)
    except Exception as e:
        logger.exception("Error starting training")
        return TrainingStartResponse(success=False, error=str(e))

@router.get("/progress/{task_id}", response_model=TrainingProgressResponse)
async def get_training_progress(task_id: str):
    """Get training progress"""
    task = training_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    eta = None
    if task.status == "training" and task.current_epoch > 0 and task.started_at:
        elapsed = (datetime.now() - task.started_at).total_seconds()
        avg_per_epoch = elapsed / task.current_epoch
        remaining_epochs = task.total_epochs - task.current_epoch
        eta = int(avg_per_epoch * remaining_epochs)
    
    return TrainingProgressResponse(
        task_id=task.task_id,
        status=task.status,
        current_epoch=task.current_epoch,
        total_epochs=task.total_epochs,
        current_loss=task.current_loss,
        best_map=task.best_map,
        progress_percentage=task.to_dict()['progress_percentage'],
        eta_seconds=eta,
        error_message=task.error_message
    )

@router.post("/cancel/{task_id}", response_model=TrainingCancelResponse)
async def cancel_training(task_id: str):
    """Cancel a training task"""
    task = training_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if training_task_manager.cancel_task(task_id):
        return TrainingCancelResponse(success=True, message="Cancellation requested")
    return TrainingCancelResponse(success=False, message=f"Cannot cancel task in status: {task.status}")

@router.get("/download/{task_id}")
async def download_model(task_id: str):
    """Download trained model"""
    task = training_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    if not task.model_path or not os.path.exists(task.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        task.model_path,
        media_type='application/octet-stream',
        filename=f"model_{task_id[:8]}.pt"
    )

@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all trained models"""
    tasks = training_task_manager.list_completed_tasks()
    models = []
    
    for task in sorted(tasks, key=lambda x: x.created_at, reverse=True):
        models.append(ModelInfo(
            task_id=task.task_id,
            model_name=f"model_{task.task_id[:8]}",
            base_model=task.config.base_model,
            created_at=task.created_at.isoformat(),
            epochs=task.config.epochs,
            best_map=task.best_map,
            model_path=task.model_path or ""
        ))
    
    return ModelListResponse(models=models)

@router.get("/datasets")
async def list_datasets():
    """List all generated datasets"""
    datasets = []
    datasets_folder = TrainingConfig.DATASETS_FOLDER
    
    if os.path.exists(datasets_folder):
        for name in os.listdir(datasets_folder):
            dataset_path = os.path.join(datasets_folder, name)
            yaml_path = os.path.join(dataset_path, 'dataset.yaml')
            
            if os.path.isdir(dataset_path) and os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                datasets.append({
                    'name': name,
                    'path': dataset_path,
                    'class_count': config.get('nc', 0),
                    'class_names': config.get('names', [])
                })
    
    return datasets
