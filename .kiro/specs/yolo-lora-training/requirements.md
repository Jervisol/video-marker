# Requirements Document

## Introduction

This document specifies the requirements for a YOLO model training and fine-tuning feature using LoRA (Low-Rank Adaptation) algorithm. The system shall enable users to create custom training datasets from video frames through an interactive annotation interface, configure training parameters, execute model fine-tuning, and download the resulting trained models.

## Glossary

- **System**: The YOLO LoRA Training Module
- **User**: A person interacting with the training interface
- **LoRA**: Low-Rank Adaptation, a parameter-efficient fine-tuning technique
- **Training Dataset**: A collection of annotated images with bounding box labels
- **Base Model**: The pre-trained YOLO model to be fine-tuned (e.g., YOLO11n)
- **Annotation**: Bounding box coordinates and class labels for objects in an image
- **Training Session**: A complete model fine-tuning process from dataset to trained model
- **Frame Capture**: A screenshot taken from a paused video frame
- **Model Checkpoint**: A saved state of the model during or after training

## Requirements

### Requirement 1

**User Story:** As a user, I want to access a dedicated training page, so that I can create and manage model training workflows separately from the detection interface.

#### Acceptance Criteria

1. WHEN a user navigates to the training page THEN the System SHALL display a new interface with video player and training controls
2. WHEN the training page loads THEN the System SHALL initialize with an empty annotation workspace
3. WHEN a user uploads a video to the training page THEN the System SHALL load the video into the player and enable frame capture controls
4. WHERE the training page is active THEN the System SHALL maintain separation from the detection interface state

### Requirement 2

**User Story:** As a user, I want to play and pause videos in the training interface, so that I can identify frames suitable for annotation.

#### Acceptance Criteria

1. WHEN a user clicks the play button THEN the System SHALL start video playback at the current position
2. WHEN a user clicks the pause button THEN the System SHALL stop video playback and maintain the current frame position
3. WHEN the video is paused THEN the System SHALL enable the frame capture button
4. WHILE the video is playing THEN the System SHALL disable the frame capture button
5. WHEN a user seeks to a different timestamp THEN the System SHALL update the video position and maintain paused state if previously paused

### Requirement 3

**User Story:** As a user, I want to capture and annotate frames from paused video, so that I can create labeled training data for specific objects.

#### Acceptance Criteria

1. WHEN a user clicks the capture button while video is paused THEN the System SHALL extract the current frame as an image
2. WHEN a frame is captured THEN the System SHALL display the image in an annotation canvas
3. WHEN a user draws a bounding box on the annotation canvas THEN the System SHALL record the box coordinates
4. WHEN a user assigns a class label to a bounding box THEN the System SHALL associate the label with the box coordinates
5. WHEN a user completes annotation THEN the System SHALL save the annotated frame with metadata including image data, bounding boxes, and class labels
6. WHEN a user annotates multiple objects in one frame THEN the System SHALL support multiple bounding boxes per image

### Requirement 4

**User Story:** As a user, I want to browse and manage saved annotated images, so that I can review my training dataset before training.

#### Acceptance Criteria

1. WHEN a user accesses the annotation gallery THEN the System SHALL display all saved annotated frames as thumbnails
2. WHEN a user clicks on a thumbnail THEN the System SHALL display the full image with visible bounding boxes and labels
3. WHEN a user selects the delete option for an annotation THEN the System SHALL remove the annotation from the dataset
4. WHEN a user views an annotation THEN the System SHALL show metadata including frame timestamp, class labels, and bounding box count
5. WHEN the annotation gallery is empty THEN the System SHALL display a message indicating no annotations exist

### Requirement 5

**User Story:** As a user, I want to select annotated images and generate a training dataset, so that I can prepare data for model fine-tuning.

#### Acceptance Criteria

1. WHEN a user selects multiple annotated images THEN the System SHALL enable the dataset generation button
2. WHEN a user clicks generate dataset THEN the System SHALL display a configuration form for training parameters
3. WHEN a user specifies dataset split ratios THEN the System SHALL validate that train, validation, and test ratios sum to one hundred percent
4. WHEN a user submits dataset configuration THEN the System SHALL convert selected annotations to YOLO format with images and label text files
5. WHEN dataset generation completes THEN the System SHALL create a dataset directory structure with train, validation, and test subdirectories
6. WHEN dataset generation completes THEN the System SHALL generate a dataset configuration YAML file with class names and paths

### Requirement 6

**User Story:** As a user, I want to configure and start model training, so that I can fine-tune YOLO models on my custom dataset.

#### Acceptance Criteria

1. WHEN a user accesses the training configuration panel THEN the System SHALL display input fields for base model selection, dataset path, epochs, batch size, and learning rate
2. WHEN a user selects a base model THEN the System SHALL validate that the model file exists in the system
3. WHEN a user specifies training parameters THEN the System SHALL validate that epochs is a positive integer, batch size is a positive integer, and learning rate is a positive decimal
4. WHEN a user clicks the start training button THEN the System SHALL initiate a background training process using the specified parameters
5. WHERE LoRA fine-tuning is enabled THEN the System SHALL apply LoRA adapters to the base model before training
6. WHEN training starts THEN the System SHALL disable the start training button and display training status

### Requirement 7

**User Story:** As a user, I want to monitor training progress in real-time, so that I can track model performance and identify issues early.

#### Acceptance Criteria

1. WHILE training is in progress THEN the System SHALL display current epoch number and total epochs
2. WHILE training is in progress THEN the System SHALL update loss metrics in real-time
3. WHILE training is in progress THEN the System SHALL display estimated time remaining
4. WHEN training completes successfully THEN the System SHALL display a completion message with final metrics
5. IF training fails THEN the System SHALL display an error message with failure reason
6. WHEN a user requests to cancel training THEN the System SHALL stop the training process and save the current checkpoint

### Requirement 8

**User Story:** As a user, I want to download trained model files, so that I can use the fine-tuned models for detection tasks.

#### Acceptance Criteria

1. WHEN training completes successfully THEN the System SHALL display a download button for the trained model
2. WHEN a user clicks the download button THEN the System SHALL provide the model file in PyTorch format with pt extension
3. WHEN a user downloads a model THEN the System SHALL include metadata file with training parameters and performance metrics
4. WHEN multiple training sessions exist THEN the System SHALL list all available trained models with timestamps
5. WHEN a trained model file is corrupted THEN the System SHALL prevent download and display an error message

### Requirement 9

**User Story:** As a user, I want the system to handle errors gracefully during annotation and training, so that I can recover from failures without losing work.

#### Acceptance Criteria

1. IF frame capture fails THEN the System SHALL display an error message and allow retry
2. IF annotation save fails THEN the System SHALL retain the annotation in memory and prompt user to retry
3. IF dataset generation fails THEN the System SHALL display detailed error information and preserve source annotations
4. IF training initialization fails THEN the System SHALL display the error reason and return to configuration state
5. WHEN the System encounters insufficient disk space THEN the System SHALL prevent operations and notify the user

### Requirement 10

**User Story:** As a user, I want to validate my annotations before training, so that I can ensure data quality and correct labeling errors.

#### Acceptance Criteria

1. WHEN a user requests annotation validation THEN the System SHALL check that all bounding boxes have associated class labels
2. WHEN a user requests annotation validation THEN the System SHALL verify that bounding box coordinates are within image boundaries
3. WHEN validation detects errors THEN the System SHALL display a list of problematic annotations with descriptions
4. WHEN a user corrects an invalid annotation THEN the System SHALL re-validate and update the validation status
5. WHEN all annotations are valid THEN the System SHALL enable the dataset generation button
