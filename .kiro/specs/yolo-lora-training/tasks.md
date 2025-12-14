# Implementation Plan: YOLO LoRA Training Module

- [x] 1. Set up project structure and configuration






  - Create directory structure for training module (annotations, datasets, models folders)
  - Extend config.py with training-specific settings (paths, limits, LoRA defaults)

  - Add required dependencies to requirements.txt (peft, hypothesis)
  - _Requirements: 1.1, 1.2_



- [x] 2. Implement data models and validation




  - [x] 2.1 Create Pydantic models for API requests and responses


    - Define FrameCaptureRequest, AnnotationRequest, DatasetRequest, TrainingRequest
    - Define response models for all endpoints
    - _Requirements: 3.5, 5.3, 6.3_
  
  - [x] 2.2 Create dataclass models for internal data structures

    - Implement BoundingBox, Annotation, TrainingConfig, TrainingTask, DatasetInfo classes
    - Add serialization/deserialization methods
    - _Requirements: 3.5, 6.1_
  
  - [ ]* 2.3 Write property test for annotation data model
    - **Property 4: Annotation save completeness**
    - **Validates: Requirements 3.5**
  
  - [ ]* 2.4 Write property test for training parameter validation
    - **Property 13: Training parameter validation**
    - **Validates: Requirements 6.2, 6.3**

- [x] 3. Implement annotation management system


  - [x] 3.1 Create AnnotationManager class


    - Implement save_annotation, get_annotation, list_annotations methods
    - Add delete_annotation and validate_annotation methods
    - Use JSON files for annotation storage with unique IDs
    - _Requirements: 3.5, 4.1, 4.3, 10.1, 10.2_
  
  - [x] 3.2 Implement annotation validation logic

    - Validate bounding box coordinates within image boundaries
    - Check all boxes have class labels
    - Return detailed validation results
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [ ]* 3.3 Write property test for annotation validation
    - **Property 22: Annotation validation completeness**
    - **Validates: Requirements 10.1, 10.2, 10.3**
  
  - [ ]* 3.4 Write property test for annotation deletion
    - **Property 7: Annotation deletion removes from dataset**
    - **Validates: Requirements 4.3**





- [x] 4. Implement dataset generation system



  - [x] 4.1 Create DatasetGenerator class

    - Implement generate_dataset method with train/val/test splitting
    - Add create_yolo_labels method for format conversion
    - Implement create_dataset_yaml for configuration file generation
    - _Requirements: 5.3, 5.4, 5.5, 5.6_
  
  - [x] 4.2 Implement YOLO format conversion

    - Convert bounding box coordinates to normalized YOLO format
    - Generate label text files with correct format
    - Handle multiple bounding boxes per image
    - _Requirements: 5.4, 3.6_
  
  - [ ]* 4.3 Write property test for YOLO format conversion
    - **Property 10: YOLO format conversion correctness**
    - **Validates: Requirements 5.4**
  
  - [ ]* 4.4 Write property test for dataset split ratios
    - **Property 9: Dataset split ratio validation**
    - **Validates: Requirements 5.3**
  
  - [ ]* 4.5 Write property test for dataset structure creation
    - **Property 11: Dataset directory structure creation**
    - **Validates: Requirements 5.5**
  
  - [x]* 4.6 Write property test for YAML generation




    - **Property 12: Dataset YAML generation**
    - **Validates: Requirements 5.6**

- [x] 5. Checkpoint - Ensure all tests pass

  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement training task management
  - [x] 6.1 Create TrainingTaskManager class
    - Implement create_task, get_task, update_progress methods
    - Add mark_completed, mark_failed, cancel_task methods
    - Use thread-safe operations with locks
    - _Requirements: 6.4, 7.1, 7.4, 7.5, 7.6_
  
  - [x] 6.2 Implement training progress tracking
    - Track current epoch, loss metrics, and estimated time
    - Support real-time progress updates
    - Handle cancellation flags
    - _Requirements: 7.1, 7.2, 7.3, 7.6_
  
  - [ ]* 6.3 Write property test for training progress monitoring
    - **Property 15: Training progress monitoring**
    - **Validates: Requirements 7.1, 7.2, 7.3**


- [x] 7. Implement LoRA training worker
  - [x] 7.1 Create lora_training_worker function
    - Load base YOLO model from file
    - Apply LoRA adapters to model layers
    - Configure training parameters (epochs, batch size, learning rate)
    - Execute training loop with progress callbacks
    - _Requirements: 6.4, 6.5, 7.1_
  
  - [x] 7.2 Implement LoRA adapter application
    - Use PEFT library to add LoRA layers
    - Configure LoRA rank, alpha, and dropout
    - Apply to convolutional and linear layers
    - _Requirements: 6.5_
  
  - [x] 7.3 Implement training checkpoint saving
    - Save model checkpoints during training
    - Save final model with LoRA weights
    - Generate metadata file with training parameters and metrics
    - _Requirements: 7.6, 8.2, 8.3_
  
  - [ ]* 7.4 Write property test for LoRA adapter application
    - **Property 14: LoRA adapter application**
    - **Validates: Requirements 6.5**
  
  - [ ]* 7.5 Write property test for training cancellation
    - **Property 18: Training cancellation checkpoint save**
    - **Validates: Requirements 7.6**

- [x] 8. Implement API endpoints for annotation management
  - [x] 8.1 Create POST /training/capture_frame endpoint
    - Extract frame from video at specified timestamp
    - Return base64 encoded image data
    - Handle video file not found errors
    - _Requirements: 3.1_
  
  - [x] 8.2 Create POST /training/annotations endpoint
    - Save annotation with bounding boxes and metadata
    - Validate annotation data before saving
    - Return annotation ID and success status
    - _Requirements: 3.5_
  
  - [x] 8.3 Create GET /training/annotations endpoint
    - List all saved annotations with thumbnails
    - Return annotation summaries with metadata
    - _Requirements: 4.1, 4.4_
  
  - [x] 8.4 Create GET /training/annotations/{annotation_id} endpoint
    - Return full annotation details with image and bounding boxes
    - Handle annotation not found errors
    - _Requirements: 4.2, 4.4_
  
  - [x] 8.5 Create DELETE /training/annotations/{annotation_id} endpoint
    - Delete annotation and associated files
    - Return success status
    - _Requirements: 4.3_
  
  - [x] 8.6 Create POST /training/annotations/validate endpoint
    - Validate selected annotations
    - Return validation results with error details
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [ ]* 8.7 Write property test for annotation metadata display
    - **Property 8: Annotation metadata completeness**
    - **Validates: Requirements 4.4**

- [x] 9. Implement API endpoints for dataset generation
  - [x] 9.1 Create POST /training/dataset/generate endpoint
    - Accept annotation IDs and split ratios
    - Validate split ratios sum to 100%
    - Generate dataset in background thread
    - Return dataset path and statistics
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [ ]* 9.2 Write property test for validation-based UI state
    - **Property 23: Validation-based UI state**
    - **Validates: Requirements 10.5**

- [x] 10. Implement API endpoints for training management
  - [x] 10.1 Create POST /training/start endpoint
    - Validate training configuration
    - Create training task
    - Submit task to background executor
    - Return task ID and initial status
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 10.2 Create GET /training/progress/{task_id} endpoint
    - Return current training progress
    - Include epoch, loss, and time estimates
    - Handle task not found errors
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 10.3 Create POST /training/cancel/{task_id} endpoint
    - Set cancellation flag for training task
    - Return cancellation status
    - _Requirements: 7.6_
  
  - [x] 10.4 Create GET /training/download/{task_id} endpoint
    - Validate model file exists and is not corrupted
    - Return model file as download
    - Include metadata file in response
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 10.5 Create GET /training/models endpoint
    - List all completed training tasks
    - Return model information with timestamps
    - Sort by creation time descending
    - _Requirements: 8.4_
  
  - [ ]* 10.6 Write property test for training completion state
    - **Property 16: Training completion state**
    - **Validates: Requirements 7.4, 8.1**
  
  - [ ]* 10.7 Write property test for training failure error reporting
    - **Property 17: Training failure error reporting**
    - **Validates: Requirements 7.5**
  
  - [ ]* 10.8 Write property test for model download format
    - **Property 19: Model download format correctness**
    - **Validates: Requirements 8.2, 8.3**
  
  - [ ]* 10.9 Write property test for model listing
    - **Property 20: Model listing with timestamps**
    - **Validates: Requirements 8.4**

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Create training page frontend
  - [x] 12.1 Create GET /training route
    - Serve training.html template
    - Pass available models and configuration
    - _Requirements: 1.1_
  
  - [x] 12.2 Create training.html template structure
    - Add video player section
    - Add annotation canvas section
    - Add annotation gallery section
    - Add training configuration panel
    - _Requirements: 1.1, 1.2, 2.1, 3.2_
  
  - [x] 12.3 Implement video player controls
    - Add play/pause buttons
    - Add seek functionality
    - Add frame capture button
    - Implement button state management based on video state
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 12.4 Write property test for video pause enables frame capture
    - **Property 1: Video pause enables frame capture**
    - **Validates: Requirements 2.3, 2.4**
  
  - [ ]* 12.5 Write property test for video seek preserves pause state
    - **Property 2: Video seek preserves pause state**
    - **Validates: Requirements 2.5**

- [x] 13. Implement annotation interface
  - [x] 13.1 Create canvas-based annotation tool
    - Implement bounding box drawing with mouse/touch
    - Add class label selection dropdown
    - Support multiple bounding boxes per frame
    - Display existing boxes with labels
    - _Requirements: 3.3, 3.4, 3.6_
  
  - [x] 13.2 Implement frame capture functionality
    - Capture current video frame to canvas
    - Convert canvas to base64 image data
    - Display captured frame in annotation canvas
    - _Requirements: 3.1, 3.2_
  
  - [x] 13.3 Implement annotation save functionality
    - Collect all bounding boxes and labels
    - Send annotation data to backend API
    - Display success/error messages
    - Clear annotation canvas after save
    - _Requirements: 3.5_
  
  - [ ]* 13.4 Write property test for frame capture extraction
    - **Property 3: Frame capture extracts current frame**
    - **Validates: Requirements 3.1, 3.2**
  
  - [ ]* 13.5 Write property test for multiple bounding boxes
    - **Property 5: Multiple bounding boxes support**
    - **Validates: Requirements 3.6**

- [x] 14. Implement annotation gallery
  - [x] 14.1 Create annotation gallery UI
    - Display annotations as thumbnail grid
    - Show metadata (timestamp, class count) on thumbnails
    - Implement thumbnail click to view full annotation
    - Add delete button for each annotation
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 14.2 Implement annotation selection
    - Add checkboxes for selecting annotations
    - Implement select all/none functionality
    - Enable dataset generation button when annotations selected
    - _Requirements: 5.1_
  
  - [ ]* 14.3 Write property test for gallery display
    - **Property 6: Gallery displays all annotations**
    - **Validates: Requirements 4.1, 4.2**

- [x] 15. Implement dataset generation interface
  - [x] 15.1 Create dataset configuration modal
    - Add input fields for train/val/test split ratios
    - Add dataset name input field
    - Validate ratios sum to 100% before submission
    - Display validation errors inline
    - _Requirements: 5.2, 5.3_
  
  - [x] 15.2 Implement dataset generation workflow
    - Send selected annotations and configuration to backend
    - Display progress indicator during generation
    - Show success message with dataset statistics
    - Handle and display generation errors
    - _Requirements: 5.4, 5.5, 5.6, 9.3_

- [x] 16. Implement training configuration interface
  - [x] 16.1 Create training configuration panel
    - Add base model selection dropdown
    - Add dataset path input (auto-filled from generation)
    - Add training parameter inputs (epochs, batch size, learning rate)
    - Add LoRA parameter inputs (rank, alpha, dropout)
    - _Requirements: 6.1_
  
  - [x] 16.2 Implement training parameter validation
    - Validate all parameters before submission
    - Display validation errors inline
    - Check model file exists
    - _Requirements: 6.2, 6.3_
  
  - [x] 16.3 Implement training start workflow
    - Send training configuration to backend
    - Disable start button during training
    - Display training status
    - _Requirements: 6.4, 6.6_

- [x] 17. Implement training progress monitoring
  - [x] 17.1 Create training progress modal
    - Display current epoch and total epochs
    - Show real-time loss metrics
    - Display estimated time remaining
    - Add progress bar visualization
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 17.2 Implement progress polling
    - Poll training progress endpoint every 2 seconds
    - Update UI with latest metrics
    - Handle training completion
    - Handle training failures with error display
    - _Requirements: 7.4, 7.5_
  
  - [x] 17.3 Implement training cancellation
    - Add cancel button to progress modal
    - Send cancellation request to backend
    - Display cancellation status
    - _Requirements: 7.6_

- [x] 18. Implement model download and management
  - [x] 18.1 Create model download functionality
    - Display download button on training completion
    - Trigger file download for model and metadata
    - Handle download errors
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 18.2 Create trained models list
    - Display all completed training tasks
    - Show model name, timestamp, and metrics
    - Add download button for each model
    - Sort by creation time descending
    - _Requirements: 8.4_

- [x] 19. Implement error handling and recovery
  - [x] 19.1 Add error handling for all operations
    - Display user-friendly error messages
    - Provide retry buttons for failed operations
    - Preserve user work on errors
    - Log errors for debugging
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 19.2 Implement disk space checking
    - Check available disk space before operations
    - Display warning when space is low
    - Prevent operations when insufficient space
    - _Requirements: 9.5_
  
  - [ ]* 19.3 Write property test for error recovery
    - **Property 21: Error recovery with retry**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

- [x] 20. Add styling and responsive design
  - [x] 20.1 Style training page components
    - Apply consistent styling with existing app
    - Make annotation canvas responsive
    - Style gallery grid layout
    - Style modals and forms
    - _Requirements: 1.1_
  
  - [x] 20.2 Implement responsive layout
    - Ensure mobile compatibility
    - Add touch support for annotation drawing
    - Optimize for different screen sizes
    - _Requirements: 1.1_

- [x] 21. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 22. Integration and documentation
  - [x] 22.1 Add navigation link to training page
    - Add link in main page navigation
    - Update index.html with training page link
    - _Requirements: 1.1_
  
  - [x] 22.2 Update README with training feature documentation
    - Document training workflow
    - Add usage examples
    - Document LoRA parameters
    - _Requirements: 1.1_
  
  - [ ]* 22.3 Write integration tests for complete workflows
    - Test annotation workflow end-to-end
    - Test dataset generation workflow
    - Test training workflow
    - _Requirements: All_
