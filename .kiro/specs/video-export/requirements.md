# Requirements Document - Video Export Feature

## Introduction

This feature adds the ability to export annotated videos with detected object bounding boxes. Users can select which object classes to include in the export, choose the output file path, and monitor the export progress. The export process runs asynchronously to avoid blocking the UI, and users can cancel the operation at any time.

## Glossary

- **Export System**: The backend service that processes video frames and generates annotated output video
- **Export Button**: UI control that initiates the video export workflow
- **File Dialog**: Browser-based interface for selecting the output file path and name
- **Progress Bar**: Visual indicator showing export completion percentage based on processed frames
- **Export Task**: An asynchronous background process that handles video annotation and file generation
- **Cancel Button**: UI control that terminates an ongoing export task and releases resources
- **Annotated Video**: Output video file with bounding boxes drawn around detected objects
- **Frame Processing**: The operation of detecting objects and drawing bounding boxes on a single video frame
- **Export Worker**: Background thread or process that executes the export task without blocking the main application

## Requirements

### Requirement 1

**User Story:** As a user, I want to export an annotated video with detected objects, so that I can save and share the detection results.

#### Acceptance Criteria

1. WHEN the user clicks the export button THEN the system SHALL display a file dialog for selecting the output path and filename
2. WHEN the user confirms the file path THEN the system SHALL begin processing the video with the currently selected object classes
3. WHEN processing each frame THEN the system SHALL detect objects and draw bounding boxes with class labels and confidence scores
4. WHEN all frames are processed THEN the system SHALL save the annotated video to the specified file path
5. WHEN the export completes successfully THEN the system SHALL notify the user and provide access to the output file

### Requirement 2

**User Story:** As a user, I want to see the export progress, so that I know how long the operation will take and that the system is working.

#### Acceptance Criteria

1. WHEN the export task starts THEN the system SHALL display a progress bar showing 0% completion
2. WHILE frames are being processed THEN the system SHALL update the progress bar based on the ratio of processed frames to total frames
3. WHEN the progress bar updates THEN the system SHALL display the percentage as a numeric value
4. WHEN the export completes THEN the system SHALL show 100% completion before closing the progress indicator
5. WHEN the export is cancelled THEN the system SHALL stop updating the progress bar and display a cancellation message

### Requirement 3

**User Story:** As a user, I want to cancel an ongoing export operation, so that I can stop the process if I change my mind or need to make adjustments.

#### Acceptance Criteria

1. WHILE an export is in progress THEN the system SHALL display a cancel button
2. WHEN the user clicks the cancel button THEN the system SHALL terminate the export task immediately
3. WHEN the export is cancelled THEN the system SHALL release all resources including file handles and memory buffers
4. WHEN the export is cancelled THEN the system SHALL delete any partially created output file
5. WHEN cancellation completes THEN the system SHALL return the UI to its normal state and allow new operations

### Requirement 4

**User Story:** As a developer, I want the export process to run in a background thread or process, so that the UI remains responsive during long export operations.

#### Acceptance Criteria

1. WHEN the export task starts THEN the system SHALL execute the processing in a separate thread or process
2. WHILE the export is running THEN the system SHALL allow the user to interact with other UI elements
3. WHEN the background task processes frames THEN the system SHALL send progress updates to the main thread
4. WHEN the background task completes THEN the system SHALL notify the main thread of success or failure
5. WHEN the application closes WHILE an export is running THEN the system SHALL terminate the background task gracefully

### Requirement 5

**User Story:** As a user, I want the export to use only the object classes I have selected, so that the output video shows only the detections I'm interested in.

#### Acceptance Criteria

1. WHEN the export starts THEN the system SHALL use the currently selected classes from the checkbox list
2. WHEN processing frames THEN the system SHALL filter detection results to include only the selected classes
3. WHEN no classes are selected THEN the system SHALL prevent the export operation and display an error message
4. WHEN drawing bounding boxes THEN the system SHALL use the same visual style as the real-time detection display
5. WHEN multiple objects of selected classes are detected THEN the system SHALL draw bounding boxes for all of them

### Requirement 6

**User Story:** As a developer, I want to implement a backend API endpoint for video export, so that the frontend can request annotated video generation.

#### Acceptance Criteria

1. WHEN the backend receives an export request THEN the system SHALL validate the video path and selected classes
2. WHEN starting the export THEN the system SHALL create a unique task identifier and return it to the frontend
3. WHEN processing frames THEN the system SHALL apply the YOLO model and draw bounding boxes on each frame
4. WHEN writing the output video THEN the system SHALL maintain the original video's resolution, frame rate, and codec settings
5. WHEN the export completes THEN the system SHALL save the file and update the task status to completed

### Requirement 7

**User Story:** As a developer, I want to implement progress tracking for export tasks, so that the frontend can display accurate progress information.

#### Acceptance Criteria

1. WHEN an export task is created THEN the system SHALL initialize a progress tracker with total frame count
2. WHEN each frame is processed THEN the system SHALL increment the processed frame counter
3. WHEN the frontend requests progress THEN the system SHALL return the current processed frame count and total frame count
4. WHEN calculating progress percentage THEN the system SHALL use the formula: (processed_frames / total_frames) * 100
5. WHEN the task completes or is cancelled THEN the system SHALL clean up the progress tracker

### Requirement 8

**User Story:** As a developer, I want to implement task cancellation, so that users can stop export operations and free system resources.

#### Acceptance Criteria

1. WHEN a cancellation request is received THEN the system SHALL set a cancellation flag for the export task
2. WHEN the export worker checks the cancellation flag THEN the system SHALL stop processing immediately if cancellation is requested
3. WHEN stopping the export THEN the system SHALL close all file handles and release video capture resources
4. WHEN the task is cancelled THEN the system SHALL delete the incomplete output file if it exists
5. WHEN cancellation completes THEN the system SHALL update the task status to cancelled and notify the frontend

### Requirement 9

**User Story:** As a user, I want the export button to be easily accessible, so that I can quickly initiate an export when needed.

#### Acceptance Criteria

1. WHEN the page loads with a video THEN the system SHALL display the export button below the class selection list
2. WHEN no video is loaded THEN the system SHALL disable or hide the export button
3. WHEN an export is already in progress THEN the system SHALL disable the export button to prevent multiple simultaneous exports
4. WHEN the export completes or is cancelled THEN the system SHALL re-enable the export button
5. WHEN hovering over the export button THEN the system SHALL display a tooltip explaining its function

### Requirement 10

**User Story:** As a user, I want clear error messages if the export fails, so that I can understand what went wrong and how to fix it.

#### Acceptance Criteria

1. WHEN the export fails due to file permission errors THEN the system SHALL display a message indicating the file path is not writable
2. WHEN the export fails due to insufficient disk space THEN the system SHALL display a message indicating storage is full
3. WHEN the export fails due to video codec errors THEN the system SHALL display a message suggesting alternative output formats
4. WHEN the export fails due to model errors THEN the system SHALL display a message indicating detection processing failed
5. WHEN any error occurs THEN the system SHALL log detailed error information for debugging purposes

