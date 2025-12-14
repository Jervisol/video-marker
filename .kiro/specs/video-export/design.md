# Design Document - Video Export Feature

## Overview

The video export feature enables users to generate annotated videos with object detection bounding boxes. The system uses a client-server architecture where the frontend initiates export requests and monitors progress, while the backend handles video processing asynchronously using background workers. The design prioritizes non-blocking operations, real-time progress updates, and graceful cancellation.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Browser)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Export Button│  │ Progress Bar │  │ Cancel Button│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────┼─────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌─────────────────────────┼─────────────────────────────┐  │
│  │         API Layer       │                             │  │
│  │  ┌──────────────────────▼──────────────────────┐     │  │
│  │  │ POST /export/start                          │     │  │
│  │  │ GET  /export/progress/{task_id}             │     │  │
│  │  │ POST /export/cancel/{task_id}               │     │  │
│  │  └──────────────────────┬──────────────────────┘     │  │
│  └─────────────────────────┼─────────────────────────────┘  │
│                             │                                │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │         Task Manager                                  │  │
│  │  - Create export tasks                                │  │
│  │  - Track task status                                  │  │
│  │  - Manage task lifecycle                              │  │
│  └─────────────────────────┬─────────────────────────────┘  │
│                             │                                │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │      Export Worker (ThreadPoolExecutor/Process)       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │ Frame Reader │→ │ YOLO Detector│→ │ Frame Writer│ │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Export Initiation**
   - User clicks export button → File dialog opens
   - User selects path → Frontend sends POST /export/start
   - Backend creates task → Returns task_id
   - Frontend starts polling for progress

2. **Progress Monitoring**
   - Frontend polls GET /export/progress/{task_id} every 500ms
   - Backend returns {processed_frames, total_frames, status}
   - Frontend updates progress bar

3. **Cancellation**
   - User clicks cancel → Frontend sends POST /export/cancel/{task_id}
   - Backend sets cancellation flag
   - Worker checks flag and stops processing
   - Resources are released

## Components and Interfaces

### Frontend Components

#### 1. ExportButton Component

**Location:** Below the class selection grid in `.controls-section`

**HTML Structure:**
```html
<div class="export-section">
    <button id="export-btn" class="export-button" onclick="initiateExport()">
        <span class="export-icon">📥</span>
        导出标注视频
    </button>
</div>
```

**State:**
- `isExporting`: boolean - Whether an export is in progress
- `currentTaskId`: string | null - Active export task ID

**Methods:**
- `initiateExport()`: Opens file dialog and starts export
- `enableExportButton()`: Enables the button
- `disableExportButton()`: Disables the button during export

#### 2. ProgressModal Component

**HTML Structure:**
```html
<div id="export-modal" class="modal">
    <div class="modal-content">
        <h3>导出进度</h3>
        <div class="progress-container">
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill"></div>
            </div>
            <div class="progress-text">
                <span id="progress-percentage">0%</span>
                <span id="progress-frames">0 / 0 帧</span>
            </div>
        </div>
        <div class="modal-actions">
            <button id="cancel-export-btn" onclick="cancelExport()">取消</button>
        </div>
    </div>
</div>
```

**State:**
- `isVisible`: boolean - Modal visibility
- `progress`: number - Current progress percentage (0-100)
- `processedFrames`: number - Frames processed so far
- `totalFrames`: number - Total frames to process

**Methods:**
- `show()`: Display the modal
- `hide()`: Hide the modal
- `updateProgress(processed, total)`: Update progress display
- `showError(message)`: Display error message
- `showSuccess()`: Display completion message

#### 3. Export Manager (JavaScript)

**Responsibilities:**
- Coordinate export workflow
- Poll backend for progress updates
- Handle cancellation requests
- Manage UI state

**Key Functions:**
```javascript
async function initiateExport() {
    // 1. Get selected classes
    // 2. Show file save dialog (using File System Access API or fallback)
    // 3. Send export request to backend
    // 4. Start progress polling
    // 5. Show progress modal
}

async function pollProgress(taskId) {
    // Poll /export/progress/{taskId} every 500ms
    // Update progress bar
    // Check for completion or errors
}

async function cancelExport() {
    // Send cancellation request
    // Stop polling
    // Hide modal
}
```

### Backend Components

#### 1. Export API Endpoints

**POST /export/start**

Request:
```json
{
    "video_filename": "video.mp4",
    "selected_classes": ["person", "car", "bicycle"],
    "selected_model": "yolo11n",
    "output_filename": "annotated_video.mp4"
}
```

Response:
```json
{
    "task_id": "uuid-string",
    "status": "started",
    "total_frames": 1500
}
```

**GET /export/progress/{task_id}**

Response:
```json
{
    "task_id": "uuid-string",
    "status": "processing",  // "processing" | "completed" | "cancelled" | "failed"
    "processed_frames": 750,
    "total_frames": 1500,
    "progress_percentage": 50.0,
    "error_message": null
}
```

**POST /export/cancel/{task_id}**

Response:
```json
{
    "task_id": "uuid-string",
    "status": "cancelled",
    "message": "Export cancelled successfully"
}
```

#### 2. ExportTaskManager

**Responsibilities:**
- Create and track export tasks
- Store task metadata and status
- Provide task status queries
- Clean up completed tasks

**Data Structure:**
```python
class ExportTask:
    task_id: str
    video_path: str
    output_path: str
    selected_classes: List[str]
    selected_model: str
    status: str  # "pending", "processing", "completed", "cancelled", "failed"
    processed_frames: int
    total_frames: int
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    cancellation_flag: threading.Event
```

**Methods:**
```python
def create_task(video_path, output_path, selected_classes, selected_model) -> ExportTask
def get_task(task_id) -> Optional[ExportTask]
def update_progress(task_id, processed_frames)
def mark_completed(task_id)
def mark_failed(task_id, error_message)
def cancel_task(task_id)
def cleanup_old_tasks()
```

#### 3. ExportWorker

**Responsibilities:**
- Process video frames in background
- Apply YOLO detection
- Draw bounding boxes
- Write annotated frames to output video
- Report progress
- Handle cancellation

**Implementation:**
```python
def export_worker(task: ExportTask, model: YOLO):
    try:
        # Open input video
        cap = cv2.VideoCapture(task.video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(task.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            # Check cancellation flag
            if task.cancellation_flag.is_set():
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame)
            
            # Draw bounding boxes
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    if class_name in task.selected_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            
            # Update progress
            frame_count += 1
            task_manager.update_progress(task.task_id, frame_count)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Mark as completed or cancelled
        if task.cancellation_flag.is_set():
            os.remove(task.output_path)  # Delete incomplete file
            task_manager.mark_cancelled(task.task_id)
        else:
            task_manager.mark_completed(task.task_id)
            
    except Exception as e:
        task_manager.mark_failed(task.task_id, str(e))
        if os.path.exists(task.output_path):
            os.remove(task.output_path)
```

## Data Models

### ExportTask Model

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import threading

@dataclass
class ExportTask:
    task_id: str
    video_path: str
    output_path: str
    selected_classes: List[str]
    selected_model: str
    status: str = "pending"
    processed_frames: int = 0
    total_frames: int = 0
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    cancellation_flag: threading.Event = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.cancellation_flag is None:
            self.cancellation_flag = threading.Event()
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "status": self.status,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "progress_percentage": (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0,
            "error_message": self.error_message
        }
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Export task creation returns valid task ID

*For any* valid export request with video path and selected classes, creating an export task should return a unique task ID that can be used to query task status.

**Validates: Requirements 1.2, 6.2**

### Property 2: Progress monotonically increases

*For any* export task, the processed frame count should never decrease during the task lifecycle, ensuring progress only moves forward.

**Validates: Requirements 2.2, 7.2**

### Property 3: Progress percentage bounds

*For any* export task, the progress percentage should always be between 0 and 100 inclusive, calculated as (processed_frames / total_frames) * 100.

**Validates: Requirements 2.3, 7.4**

### Property 4: Cancellation stops processing

*For any* export task, when cancellation is requested, the worker should stop processing within one frame interval and set the task status to "cancelled".

**Validates: Requirements 3.2, 8.2**

### Property 5: Resource cleanup on cancellation

*For any* cancelled export task, all resources (file handles, video capture objects, memory buffers) should be released and any partial output file should be deleted.

**Validates: Requirements 3.3, 3.4, 8.4**

### Property 6: Only selected classes are annotated

*For any* export task with selected classes, the output video should only contain bounding boxes for objects belonging to those classes, filtering out all other detections.

**Validates: Requirements 5.1, 5.2**

### Property 7: Output video preserves input properties

*For any* export task, the output video should maintain the same resolution, frame rate, and total frame count as the input video.

**Validates: Requirements 6.4**

### Property 8: Task status transitions are valid

*For any* export task, status transitions should follow the valid state machine: pending → processing → (completed | cancelled | failed), with no invalid transitions.

**Validates: Requirements 6.5, 7.5**

### Property 9: Progress updates are atomic

*For any* concurrent progress queries, the returned processed frame count and total frame count should be consistent (no race conditions).

**Validates: Requirements 7.1, 7.3**

### Property 10: Export button state consistency

*For any* UI state, the export button should be enabled if and only if: (1) a video is loaded, (2) at least one class is selected, and (3) no export is currently in progress.

**Validates: Requirements 9.2, 9.3, 9.4**

## Error Handling

### Frontend Error Handling

1. **No Classes Selected**
   - Check before initiating export
   - Display alert: "请至少选择一个要检测的物品类别"

2. **File Dialog Cancelled**
   - User cancels file selection
   - No action needed, return to normal state

3. **Network Errors**
   - Retry progress polling up to 3 times
   - Display error if backend unreachable
   - Allow user to dismiss and retry

4. **Backend Errors**
   - Display error message from backend response
   - Provide option to retry or cancel

### Backend Error Handling

1. **Invalid Video Path**
   - Return 404 error with message
   - Log error details

2. **File Permission Errors**
   - Return 403 error with message
   - Suggest checking file permissions

3. **Insufficient Disk Space**
   - Catch IOError during write
   - Return 507 error with message
   - Clean up partial files

4. **Video Codec Errors**
   - Catch cv2 exceptions
   - Return 500 error with codec details
   - Suggest alternative formats

5. **Model Inference Errors**
   - Catch YOLO exceptions
   - Mark task as failed
   - Log full stack trace

## Testing Strategy

### Unit Tests

1. **ExportTaskManager Tests**
   - Test task creation and ID generation
   - Test task status updates
   - Test task retrieval
   - Test cancellation flag setting

2. **Progress Calculation Tests**
   - Test progress percentage calculation
   - Test edge cases (0 frames, 1 frame)
   - Test rounding behavior

3. **File Path Validation Tests**
   - Test valid and invalid paths
   - Test path sanitization
   - Test filename generation

### Integration Tests

1. **Export Workflow Test**
   - Create export task
   - Verify task ID returned
   - Poll progress until completion
   - Verify output file exists

2. **Cancellation Test**
   - Start export task
   - Cancel after processing some frames
   - Verify task status is "cancelled"
   - Verify partial file is deleted

3. **Error Handling Test**
   - Test with invalid video path
   - Test with unwritable output path
   - Verify appropriate error responses

### Property-Based Tests

Tests will be implemented using the property-based testing framework specified in the requirements, validating the correctness properties defined above.

## Performance Considerations

1. **Background Processing**
   - Use ThreadPoolExecutor with max 2 workers
   - Prevents blocking main application thread
   - Limits concurrent exports to avoid resource exhaustion

2. **Progress Polling**
   - Poll every 500ms (balance between responsiveness and overhead)
   - Use lightweight GET requests
   - Cache task status in memory

3. **Video Processing**
   - Process frames sequentially (no parallel frame processing)
   - Use efficient cv2 operations
   - Minimize memory allocations

4. **Resource Limits**
   - Limit concurrent export tasks to 2
   - Implement task queue if limit exceeded
   - Clean up completed tasks after 1 hour

## Security Considerations

1. **Path Traversal Prevention**
   - Validate output paths
   - Restrict to allowed directories
   - Sanitize filenames

2. **Resource Limits**
   - Limit maximum video file size (e.g., 2GB)
   - Limit maximum video duration (e.g., 1 hour)
   - Implement timeout for stuck tasks

3. **Task ID Security**
   - Use UUID4 for unpredictable task IDs
   - Validate task ID format in API requests
   - Prevent task ID enumeration

