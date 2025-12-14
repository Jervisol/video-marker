# Design Document: YOLO LoRA Training Module

## Overview

The YOLO LoRA Training Module is a comprehensive web-based interface that enables users to create custom training datasets from video frames and fine-tune YOLO models using LoRA (Low-Rank Adaptation) for parameter-efficient transfer learning. The system integrates with the existing video detection application, providing a dedicated training workflow that includes video playback, interactive frame annotation, dataset management, training configuration, and model export capabilities.

The module follows the existing FastAPI architecture pattern with async endpoints, background task processing using ThreadPoolExecutor, and a responsive web interface. It leverages Ultralytics YOLO's built-in training capabilities while adding LoRA fine-tuning support for efficient model customization.

## Architecture

### System Components

The training module consists of four primary layers:

1. **Presentation Layer**: HTML/CSS/JavaScript interface for user interaction
2. **API Layer**: FastAPI endpoints for handling training-related requests
3. **Business Logic Layer**: Training orchestration, dataset generation, and annotation management
4. **Storage Layer**: File system storage for annotations, datasets, and trained models

### Integration Points

- **Existing Video Player**: Reuses video upload and playback infrastructure
- **Model Registry**: Integrates with the existing `trained_models` dictionary
- **Static File Serving**: Uses existing `/static` mount for storing training artifacts
- **Configuration**: Extends `config.py` with training-specific settings

### Technology Stack

- **Backend**: FastAPI with async/await patterns
- **Training Framework**: Ultralytics YOLO with LoRA adapters
- **LoRA Implementation**: PEFT (Parameter-Efficient Fine-Tuning) library or custom LoRA layers
- **Image Processing**: OpenCV, PIL/Pillow for frame extraction and annotation
- **Data Format**: YOLO format (images + txt label files) with YAML configuration
- **Concurrency**: ThreadPoolExecutor for background training tasks
- **Frontend**: Vanilla JavaScript with Canvas API for annotation interface

## Components and Interfaces

### 1. Training Page Route

**Endpoint**: `GET /training`

**Purpose**: Serve the dedicated training interface

**Template**: `templates/training.html`

**Context Data**:
```python
{
    'request': Request,
    'trained_models': List[str],  # Available base models
    'default_model': str
}
```

### 2. Annotation Manager

**Class**: `AnnotationManager`

**Responsibilities**:
- Store and retrieve frame annotations
- Validate annotation data
- Generate unique annotation IDs
- Manage annotation metadata

**Methods**:
```python
class AnnotationManager:
    def __init__(self, storage_path: str)
    def save_annotation(self, annotation: Annotation) -> str
    def get_annotation(self, annotation_id: str) -> Optional[Annotation]
    def list_annotations(self) -> List[Annotation]
    def delete_annotation(self, annotation_id: str) -> bool
    def validate_annotation(self, annotation: Annotation) -> ValidationResult
```

### 3. Dataset Generator

**Class**: `DatasetGenerator`

**Responsibilities**:
- Convert annotations to YOLO format
- Split data into train/val/test sets
- Generate dataset YAML configuration
- Create directory structure

**Methods**:
```python
class DatasetGenerator:
    def generate_dataset(
        self,
        annotations: List[Annotation],
        output_path: str,
        split_ratios: Dict[str, float],
        class_names: List[str]
    ) -> DatasetInfo
    
    def create_yolo_labels(self, annotation: Annotation) -> str
    def create_dataset_yaml(self, dataset_info: DatasetInfo) -> str
```

### 4. Training Task Manager

**Class**: `TrainingTaskManager`

**Responsibilities**:
- Manage training task lifecycle
- Track training progress
- Handle task cancellation
- Store training metrics

**Methods**:
```python
class TrainingTaskManager:
    def create_task(self, config: TrainingConfig) -> TrainingTask
    def get_task(self, task_id: str) -> Optional[TrainingTask]
    def update_progress(self, task_id: str, metrics: TrainingMetrics)
    def mark_completed(self, task_id: str, model_path: str)
    def mark_failed(self, task_id: str, error: str)
    def cancel_task(self, task_id: str) -> bool
```

### 5. LoRA Training Worker

**Function**: `lora_training_worker(task: TrainingTask)`

**Purpose**: Execute model fine-tuning in background thread

**Process**:
1. Load base YOLO model
2. Apply LoRA adapters to model layers
3. Configure training parameters
4. Execute training loop with progress callbacks
5. Save fine-tuned model with LoRA weights
6. Generate training report

### 6. API Endpoints

#### Frame Capture
```python
@app.post('/training/capture_frame')
async def capture_frame(request: FrameCaptureRequest) -> FrameCaptureResponse
```

#### Save Annotation
```python
@app.post('/training/annotations')
async def save_annotation(request: AnnotationRequest) -> AnnotationResponse
```

#### List Annotations
```python
@app.get('/training/annotations')
async def list_annotations() -> List[AnnotationSummary]
```

#### Get Annotation Detail
```python
@app.get('/training/annotations/{annotation_id}')
async def get_annotation(annotation_id: str) -> AnnotationDetail
```

#### Delete Annotation
```python
@app.delete('/training/annotations/{annotation_id}')
async def delete_annotation(annotation_id: str) -> DeleteResponse
```

#### Validate Annotations
```python
@app.post('/training/annotations/validate')
async def validate_annotations(request: ValidationRequest) -> ValidationResponse
```

#### Generate Dataset
```python
@app.post('/training/dataset/generate')
async def generate_dataset(request: DatasetRequest) -> DatasetResponse
```

#### Start Training
```python
@app.post('/training/start')
async def start_training(request: TrainingRequest) -> TrainingStartResponse
```

#### Training Progress
```python
@app.get('/training/progress/{task_id}')
async def get_training_progress(task_id: str) -> TrainingProgressResponse
```

#### Cancel Training
```python
@app.post('/training/cancel/{task_id}')
async def cancel_training(task_id: str) -> CancelResponse
```

#### Download Model
```python
@app.get('/training/download/{task_id}')
async def download_model(task_id: str) -> FileResponse
```

## Data Models

### Annotation
```python
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    class_id: int

@dataclass
class Annotation:
    annotation_id: str
    video_filename: str
    frame_timestamp: float
    frame_number: int
    image_data: str  # base64 encoded
    bounding_boxes: List[BoundingBox]
    image_width: int
    image_height: int
    created_at: datetime
```

### Training Configuration
```python
@dataclass
class TrainingConfig:
    base_model: str
    dataset_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    image_size: int = 640
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Training Task
```python
@dataclass
class TrainingTask:
    task_id: str
    config: TrainingConfig
    status: str  # pending, training, completed, failed, cancelled
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    best_map: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    cancellation_flag: threading.Event
```

### Dataset Information
```python
@dataclass
class DatasetInfo:
    dataset_path: str
    train_count: int
    val_count: int
    test_count: int
    class_names: List[str]
    yaml_path: str
```

### Pydantic Request/Response Models

```python
class FrameCaptureRequest(BaseModel):
    video_filename: str
    timestamp: float

class AnnotationRequest(BaseModel):
    video_filename: str
    frame_timestamp: float
    frame_number: int
    image_data: str
    bounding_boxes: List[Dict]
    image_width: int
    image_height: int

class DatasetRequest(BaseModel):
    annotation_ids: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    dataset_name: str

class TrainingRequest(BaseModel):
    base_model: str
    dataset_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_rank: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.1

class TrainingProgressResponse(BaseModel):
    task_id: str
    status: str
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_map: float
    progress_percentage: float
    error_message: Optional[str] = None
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

Before defining the final correctness properties, I've analyzed the prework to identify and eliminate redundancy:

**Redundancies Identified:**
- Properties 2.3 and 2.4 (pause/play button states) can be combined into a single property about button state consistency
- Properties 3.3 and 3.4 (recording coordinates and associating labels) are both covered by property 3.5 (complete save with all metadata)
- Properties 4.1 and 4.2 (gallery display and thumbnail click) can be combined into a comprehensive gallery interaction property
- Properties 7.1, 7.2, and 7.3 (progress display elements) can be combined into a single comprehensive progress monitoring property
- Properties 8.1 and 8.2 (download button and file format) can be combined into a single download functionality property

**Properties to Keep:**
After reflection, the following properties provide unique validation value without redundancy.

### Correctness Properties

Property 1: Video pause enables frame capture
*For any* video player state, when the video is paused, the frame capture button should be enabled, and when the video is playing, the frame capture button should be disabled
**Validates: Requirements 2.3, 2.4**

Property 2: Video seek preserves pause state
*For any* video in a paused state, seeking to a different timestamp should maintain the paused state
**Validates: Requirements 2.5**

Property 3: Frame capture extracts current frame
*For any* paused video at any timestamp, clicking the capture button should extract the current frame and display it in the annotation canvas
**Validates: Requirements 3.1, 3.2**

Property 4: Annotation save completeness
*For any* completed annotation with bounding boxes and class labels, saving should persist all metadata including image data, bounding box coordinates, class labels, frame timestamp, and frame number
**Validates: Requirements 3.5**

Property 5: Multiple bounding boxes support
*For any* frame annotation, the system should support adding multiple bounding boxes with different class labels
**Validates: Requirements 3.6**

Property 6: Gallery displays all annotations
*For any* set of saved annotations, the gallery should display all annotations as thumbnails, and clicking any thumbnail should show the full image with all bounding boxes and labels visible
**Validates: Requirements 4.1, 4.2**

Property 7: Annotation deletion removes from dataset
*For any* annotation in the dataset, deleting it should remove it completely from the gallery and make it unavailable for dataset generation
**Validates: Requirements 4.3**

Property 8: Annotation metadata completeness
*For any* annotation viewed in the gallery, all metadata including frame timestamp, class labels, and bounding box count should be displayed
**Validates: Requirements 4.4**

Property 9: Dataset split ratio validation
*For any* set of dataset split ratios (train, validation, test), the system should validate that they sum to exactly 100 percent before allowing dataset generation
**Validates: Requirements 5.3**

Property 10: YOLO format conversion correctness
*For any* annotation with bounding boxes, converting to YOLO format should produce a text file where each line contains normalized coordinates (class_id, x_center, y_center, width, height) with values between 0 and 1
**Validates: Requirements 5.4**

Property 11: Dataset directory structure creation
*For any* dataset generation request, the system should create a directory structure containing train, val, and test subdirectories, each with images and labels folders
**Validates: Requirements 5.5**

Property 12: Dataset YAML generation
*For any* generated dataset, a YAML configuration file should be created containing the correct number of classes, class names list, and paths to train, val, and test directories
**Validates: Requirements 5.6**

Property 13: Training parameter validation
*For any* training configuration, the system should validate that epochs is a positive integer, batch size is a positive integer, learning rate is a positive decimal, and the selected base model file exists
**Validates: Requirements 6.2, 6.3**

Property 14: LoRA adapter application
*For any* training configuration with LoRA enabled, the system should apply LoRA adapters to the base model's convolutional and linear layers before training begins
**Validates: Requirements 6.5**

Property 15: Training progress monitoring
*For any* training task in progress, the system should continuously update and display current epoch, total epochs, current loss, and estimated time remaining
**Validates: Requirements 7.1, 7.2, 7.3**

Property 16: Training completion state
*For any* training task that completes successfully, the system should display a completion message with final metrics and enable the model download button
**Validates: Requirements 7.4, 8.1**

Property 17: Training failure error reporting
*For any* training task that fails, the system should display an error message containing the specific failure reason
**Validates: Requirements 7.5**

Property 18: Training cancellation checkpoint save
*For any* training task in progress, when cancellation is requested, the system should stop training and save the current model checkpoint before marking the task as cancelled
**Validates: Requirements 7.6**

Property 19: Model download format correctness
*For any* successfully trained model, downloading should provide a PyTorch .pt file and an accompanying metadata JSON file containing training parameters and performance metrics
**Validates: Requirements 8.2, 8.3**

Property 20: Model listing with timestamps
*For any* set of completed training tasks, the system should list all trained models with their creation timestamps in descending order
**Validates: Requirements 8.4**

Property 21: Error recovery with retry
*For any* failed operation (frame capture, annotation save, dataset generation, training initialization), the system should display an error message and provide a mechanism to retry the operation
**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

Property 22: Annotation validation completeness
*For any* set of annotations, validation should check that all bounding boxes have class labels and all coordinates are within image boundaries, returning a list of any violations
**Validates: Requirements 10.1, 10.2, 10.3**

Property 23: Validation-based UI state
*For any* set of annotations, the dataset generation button should be enabled if and only if all annotations pass validation
**Validates: Requirements 10.5**

## Error Handling

### Client-Side Errors

1. **Invalid User Input**
   - Validate form inputs before submission
   - Display inline error messages for invalid fields
   - Prevent submission until all validations pass

2. **Network Failures**
   - Implement retry logic with exponential backoff
   - Display user-friendly error messages
   - Preserve user work in browser storage

3. **Canvas Drawing Errors**
   - Validate bounding box coordinates
   - Prevent boxes outside image boundaries
   - Handle touch and mouse events consistently

### Server-Side Errors

1. **File System Errors**
   - Check disk space before operations
   - Handle permission errors gracefully
   - Implement atomic file operations where possible

2. **Model Loading Errors**
   - Validate model file integrity
   - Provide fallback to default models
   - Log detailed error information

3. **Training Errors**
   - Catch CUDA out-of-memory errors
   - Handle dataset loading failures
   - Implement training checkpoints for recovery

4. **Concurrent Access**
   - Use file locking for shared resources
   - Implement thread-safe task management
   - Handle race conditions in annotation saves

### Error Response Format

All API errors follow a consistent format:
```python
{
    "error": "Error category",
    "message": "User-friendly error description",
    "details": "Technical details for debugging",
    "retry_possible": bool
}
```

## Testing Strategy

### Unit Testing

**Framework**: pytest

**Coverage Areas**:
- Annotation validation logic
- YOLO format conversion functions
- Dataset split calculations
- Bounding box coordinate transformations
- YAML configuration generation
- Training parameter validation

**Example Tests**:
```python
def test_yolo_format_conversion():
    """Test that bounding boxes are correctly converted to YOLO format"""
    annotation = create_test_annotation()
    yolo_label = convert_to_yolo_format(annotation)
    assert all(0 <= coord <= 1 for coord in yolo_label.coordinates)

def test_dataset_split_ratios():
    """Test that dataset split ratios sum to 100%"""
    with pytest.raises(ValidationError):
        validate_split_ratios(train=0.7, val=0.2, test=0.2)
```

### Property-Based Testing

**Framework**: Hypothesis (Python property-based testing library)

**Configuration**: Each property test should run a minimum of 100 iterations

**Test Tagging**: Each property-based test must include a comment with the format:
`# Feature: yolo-lora-training, Property {number}: {property_text}`

**Coverage Areas**:

1. **Annotation Properties**
   - Generate random annotations with various bounding box configurations
   - Test save/load round-trip consistency
   - Verify metadata preservation

2. **Dataset Generation Properties**
   - Generate random annotation sets
   - Test YOLO format conversion correctness
   - Verify directory structure creation

3. **Validation Properties**
   - Generate random valid and invalid annotations
   - Test validation logic completeness
   - Verify error detection accuracy

4. **Training Configuration Properties**
   - Generate random training parameters
   - Test parameter validation
   - Verify LoRA adapter application

**Example Property Tests**:
```python
from hypothesis import given, strategies as st

@given(
    x1=st.integers(min_value=0, max_value=1920),
    y1=st.integers(min_value=0, max_value=1080),
    width=st.integers(min_value=1, max_value=500),
    height=st.integers(min_value=1, max_value=500)
)
def test_property_yolo_format_normalization(x1, y1, width, height):
    """
    Feature: yolo-lora-training, Property 10: YOLO format conversion correctness
    
    For any bounding box coordinates, YOLO format conversion should produce
    normalized values between 0 and 1.
    """
    image_width, image_height = 1920, 1080
    x2, y2 = x1 + width, y1 + height
    
    # Ensure box is within image
    x2 = min(x2, image_width)
    y2 = min(y2, image_height)
    
    yolo_coords = convert_to_yolo_format(x1, y1, x2, y2, image_width, image_height)
    
    assert 0 <= yolo_coords.x_center <= 1
    assert 0 <= yolo_coords.y_center <= 1
    assert 0 <= yolo_coords.width <= 1
    assert 0 <= yolo_coords.height <= 1

@given(
    train_ratio=st.floats(min_value=0.1, max_value=0.9),
    val_ratio=st.floats(min_value=0.05, max_value=0.4)
)
def test_property_dataset_split_validation(train_ratio, val_ratio):
    """
    Feature: yolo-lora-training, Property 9: Dataset split ratio validation
    
    For any set of split ratios, validation should accept only when they sum to 1.0.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
        # Should pass validation
        result = validate_split_ratios(train_ratio, val_ratio, test_ratio)
        assert result.is_valid
    else:
        # Should fail validation
        with pytest.raises(ValidationError):
            validate_split_ratios(train_ratio, val_ratio, test_ratio)
```

### Integration Testing

**Scope**: Test complete workflows end-to-end

**Test Scenarios**:
1. Complete annotation workflow: capture → annotate → save → retrieve
2. Dataset generation workflow: select annotations → configure → generate → verify
3. Training workflow: configure → start → monitor → complete → download
4. Error recovery: trigger errors → verify error handling → retry → succeed

### Manual Testing Checklist

- [ ] Video playback controls work correctly
- [ ] Frame capture produces clear images
- [ ] Bounding box drawing is smooth and accurate
- [ ] Annotation gallery displays correctly
- [ ] Dataset generation creates valid YOLO format
- [ ] Training progress updates in real-time
- [ ] Model download provides correct files
- [ ] Error messages are clear and actionable

## Performance Considerations

### Frame Capture Optimization
- Use canvas-based frame extraction for efficiency
- Compress images before storing (JPEG with 85% quality)
- Implement lazy loading for annotation gallery

### Dataset Generation Optimization
- Process annotations in batches
- Use multiprocessing for image conversion
- Implement progress callbacks for large datasets

### Training Optimization
- Use mixed precision training (FP16) when available
- Implement gradient accumulation for large batch sizes
- Enable CUDA optimizations for GPU training
- Use LoRA for memory-efficient fine-tuning

### Storage Optimization
- Store annotations in SQLite database for efficient queries
- Use file system for images with database references
- Implement cleanup for old training artifacts
- Compress trained models before download

## Security Considerations

### Input Validation
- Sanitize all file uploads
- Validate image dimensions and file sizes
- Prevent path traversal attacks in file operations
- Limit annotation data size to prevent DoS

### Access Control
- Implement user authentication (future enhancement)
- Restrict access to training artifacts
- Validate model file integrity before loading

### Resource Limits
- Limit concurrent training tasks
- Implement disk space quotas
- Set maximum annotation count per user
- Timeout long-running operations

## Deployment Considerations

### Dependencies
```
ultralytics>=8.0.0
torch>=2.0.0
peft>=0.5.0  # For LoRA implementation
opencv-python>=4.8.0
pillow>=10.0.0
pyyaml>=6.0
hypothesis>=6.0.0  # For property-based testing
```

### Configuration
Add to `config.py`:
```python
class TrainingConfig:
    ANNOTATIONS_FOLDER = 'static/training/annotations'
    DATASETS_FOLDER = 'static/training/datasets'
    MODELS_FOLDER = 'static/training/models'
    MAX_ANNOTATION_SIZE_MB = 10
    MAX_CONCURRENT_TRAINING = 2
    DEFAULT_LORA_RANK = 8
    DEFAULT_LORA_ALPHA = 16
```

### Directory Structure
```
static/
  training/
    annotations/
      {annotation_id}.json
      {annotation_id}.jpg
    datasets/
      {dataset_name}/
        train/
          images/
          labels/
        val/
          images/
          labels/
        test/
          images/
          labels/
        dataset.yaml
    models/
      {task_id}/
        best.pt
        last.pt
        metadata.json
```

## Future Enhancements

1. **Active Learning**: Suggest frames for annotation based on model uncertainty
2. **Collaborative Annotation**: Multi-user annotation with conflict resolution
3. **Model Versioning**: Track model lineage and performance over time
4. **Automated Augmentation**: Apply data augmentation during dataset generation
5. **Transfer Learning**: Support for other model architectures beyond YOLO
6. **Cloud Training**: Offload training to cloud GPU instances
7. **Model Comparison**: Side-by-side comparison of trained models
8. **Annotation Import/Export**: Support for COCO, Pascal VOC formats
