# Implementation Plan - Video Export Feature

## Task List

- [x] 1. Set up backend export infrastructure


  - Create ExportTask data model with all required fields
  - Implement ExportTaskManager class for task lifecycle management
  - Set up ThreadPoolExecutor for background processing
  - _Requirements: 4.1, 6.2, 7.1_



- [ ] 1.1 Create ExportTask data model
  - Define ExportTask dataclass with task_id, paths, status, progress fields
  - Add cancellation_flag as threading.Event
  - Implement to_dict() method for JSON serialization

  - _Requirements: 6.2, 7.1_

- [ ] 1.2 Implement ExportTaskManager
  - Create task storage dictionary (thread-safe)
  - Implement create_task() method with UUID generation
  - Implement get_task(), update_progress(), mark_completed() methods

  - Implement cancel_task() and cleanup_old_tasks() methods
  - _Requirements: 6.2, 7.1, 8.1_

- [x] 1.3 Set up background worker infrastructure


  - Initialize ThreadPoolExecutor with max_workers=2
  - Create worker submission mechanism
  - Implement worker exception handling
  - _Requirements: 4.1, 4.4_

- [ ] 2. Implement backend API endpoints
  - Create POST /export/start endpoint

  - Create GET /export/progress/{task_id} endpoint
  - Create POST /export/cancel/{task_id} endpoint
  - Add request validation and error handling
  - _Requirements: 6.1, 6.5, 8.1_

- [ ] 2.1 Implement POST /export/start endpoint
  - Parse request body (video_filename, selected_classes, selected_model, output_filename)

  - Validate video file exists
  - Create export task using ExportTaskManager
  - Submit task to ThreadPoolExecutor
  - Return task_id and total_frames
  - _Requirements: 1.2, 6.1, 6.2_


- [ ] 2.2 Implement GET /export/progress/{task_id} endpoint
  - Validate task_id format
  - Retrieve task from ExportTaskManager
  - Return task status, processed_frames, total_frames, progress_percentage
  - Handle task not found error
  - _Requirements: 2.2, 7.3_

- [ ] 2.3 Implement POST /export/cancel/{task_id} endpoint
  - Validate task_id format



  - Call ExportTaskManager.cancel_task()
  - Set cancellation flag
  - Return cancellation confirmation
  - _Requirements: 3.2, 8.1_

- [ ] 3. Implement export worker function
  - Create export_worker() function
  - Open input video with cv2.VideoCapture

  - Create output video with cv2.VideoWriter
  - Process frames in loop with cancellation checks
  - Apply YOLO detection and draw bounding boxes
  - Update progress after each frame
  - Handle cleanup and error cases
  - _Requirements: 1.3, 5.1, 6.3, 6.4_


- [ ] 3.1 Implement video input/output handling
  - Open input video and extract properties (fps, width, height, frame_count)
  - Create VideoWriter with matching properties
  - Implement frame reading loop
  - Handle video codec initialization
  - _Requirements: 6.4_


- [ ] 3.2 Implement frame processing with detection
  - Run YOLO model on each frame
  - Filter results by selected_classes
  - Draw bounding boxes with cv2.rectangle

  - Draw labels with class name and confidence
  - Write annotated frame to output video
  - _Requirements: 1.3, 5.1, 5.4_

- [ ] 3.3 Implement progress tracking in worker
  - Increment frame counter after each processed frame

  - Call task_manager.update_progress() periodically


  - Calculate and update progress percentage
  - _Requirements: 2.2, 7.2_

- [ ] 3.4 Implement cancellation handling in worker
  - Check cancellation_flag before processing each frame
  - Break loop if cancellation requested

  - Release video capture and writer resources
  - Delete incomplete output file on cancellation
  - _Requirements: 3.2, 3.3, 3.4, 8.2, 8.4_

- [ ] 3.5 Implement error handling in worker
  - Wrap worker in try-except block

  - Catch cv2 exceptions (codec errors, file errors)
  - Catch YOLO exceptions (model errors)
  - Mark task as failed with error message
  - Clean up partial output file on error
  - _Requirements: 10.3, 10.4_


- [ ] 4. Implement frontend export button UI
  - Add export button HTML below class selection grid
  - Style export button with icon and hover effects
  - Implement button enable/disable logic
  - Add click event handler
  - _Requirements: 9.1, 9.2, 9.3, 9.4_


- [ ] 4.1 Create export button HTML and CSS
  - Add export-section div with export button
  - Style button with primary color scheme
  - Add export icon (📥 or SVG)
  - Add hover and disabled states
  - _Requirements: 9.1, 9.5_


- [ ] 4.2 Implement export button state management
  - Create isExporting flag
  - Implement enableExportButton() function
  - Implement disableExportButton() function
  - Update button state based on video load and export status
  - _Requirements: 9.2, 9.3, 9.4_


- [ ] 5. Implement progress modal UI
  - Create modal HTML structure
  - Style modal with overlay and centered content
  - Add progress bar with fill animation
  - Add progress text (percentage and frame count)
  - Add cancel button

  - _Requirements: 2.1, 2.3, 3.1_

- [ ] 5.1 Create progress modal HTML structure
  - Add modal overlay div
  - Add modal content div with title
  - Add progress-container with progress-bar

  - Add progress-text with percentage and frame count spans
  - Add cancel button in modal-actions
  - _Requirements: 2.1, 2.3_

- [ ] 5.2 Style progress modal with CSS
  - Style modal overlay (semi-transparent background)
  - Style modal content (centered, white background, shadow)
  - Style progress bar (container and fill)

  - Style progress text (centered, readable)
  - Style cancel button (secondary color)
  - _Requirements: 2.1_

- [ ] 5.3 Implement modal show/hide functions
  - Create showProgressModal() function
  - Create hideProgressModal() function

  - Add modal visibility state management
  - Implement modal backdrop click handling (optional)
  - _Requirements: 2.1, 3.1_

- [ ] 5.4 Implement progress update function
  - Create updateProgress(processed, total) function
  - Calculate progress percentage
  - Update progress bar width

  - Update progress text (percentage and frames)
  - Animate progress bar fill
  - _Requirements: 2.2, 2.3_

- [ ] 6. Implement export workflow coordination
  - Create initiateExport() function
  - Implement file save dialog (File System Access API or fallback)

  - Send export request to backend
  - Start progress polling
  - Handle export completion and errors
  - _Requirements: 1.1, 1.2, 1.5_


- [ ] 6.1 Implement file save dialog
  - Use File System Access API (showSaveFilePicker) if available
  - Implement fallback for browsers without support
  - Set default filename with timestamp
  - Set file type filter (.mp4)
  - Handle user cancellation
  - _Requirements: 1.1_


- [ ] 6.2 Implement export request sending
  - Collect selected classes from checkboxes
  - Validate at least one class is selected
  - Get current video filename
  - Get selected model

  - Send POST request to /export/start
  - Handle response and extract task_id
  - _Requirements: 1.2, 5.1_

- [x] 6.3 Implement progress polling mechanism


  - Create pollProgress(taskId) function
  - Set up interval to poll every 500ms
  - Send GET request to /export/progress/{taskId}
  - Update progress modal with response data
  - Stop polling on completion, cancellation, or error
  - _Requirements: 2.2, 4.2_


- [ ] 6.4 Implement completion and error handling
  - Detect task completion from progress response
  - Show success message and hide modal
  - Detect task failure and show error message
  - Re-enable export button after completion/error
  - _Requirements: 1.5, 10.1, 10.2, 10.3, 10.4_


- [ ] 7. Implement cancellation workflow
  - Create cancelExport() function
  - Send cancellation request to backend
  - Stop progress polling
  - Hide progress modal
  - Re-enable export button
  - _Requirements: 3.1, 3.2, 3.5_

- [ ] 7.1 Implement cancel button handler
  - Add onclick handler to cancel button
  - Confirm cancellation with user (optional)
  - Call cancelExport() function
  - _Requirements: 3.1_

- [ ] 7.2 Implement cancellation request
  - Send POST request to /export/cancel/{taskId}
  - Handle response
  - Clear polling interval
  - Update UI state
  - _Requirements: 3.2, 3.5_

- [ ] 8. Add validation and error messages
  - Validate selected classes before export
  - Validate video is loaded
  - Display user-friendly error messages
  - Add error message styling
  - _Requirements: 5.3, 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 8.1 Implement pre-export validation
  - Check if video is loaded
  - Check if at least one class is selected
  - Show alert if validation fails
  - Prevent export if validation fails
  - _Requirements: 5.3, 9.2_

- [ ] 8.2 Implement error message display
  - Create showError(message) function
  - Display error in modal or alert
  - Style error messages (red color, icon)
  - Add dismiss button for errors
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 9. Testing and integration
  - Test export workflow end-to-end
  - Test progress updates
  - Test cancellation
  - Test error handling
  - Verify no interference with existing features
  - _Requirements: All_

- [ ] 9.1 Test complete export workflow
  - Upload video
  - Select classes
  - Click export button
  - Choose output path
  - Verify progress updates
  - Verify output file is created
  - Verify output video has correct annotations
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 9.2 Test cancellation workflow
  - Start export
  - Click cancel during processing
  - Verify task stops
  - Verify partial file is deleted
  - Verify UI returns to normal state
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 9.3 Test error scenarios
  - Test with invalid video path
  - Test with no classes selected
  - Test with unwritable output path
  - Verify appropriate error messages
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 9.4 Test concurrent operations
  - Start export
  - Verify real-time detection still works
  - Verify UI remains responsive
  - Verify no resource conflicts
  - _Requirements: 4.2, 4.3_

- [ ] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

