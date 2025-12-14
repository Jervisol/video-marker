# Implementation Plan

- [ ] 1. Create backend API endpoint for single-frame detection
  - Implement `/detect_frame` Flask route that accepts base64 encoded frames
  - Add frame decoder to convert base64 to numpy array
  - Implement detection filtering to return only selected classes
  - Return JSON response with bounding boxes, class names, and confidence scores
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ]* 1.1 Write property test for class filtering
  - **Property 4: Detection results contain only selected classes**
  - **Validates: Requirements 8.4**

- [ ]* 1.2 Write property test for error handling
  - **Property 7: Invalid frames are handled gracefully**
  - **Validates: Requirements 5.5**

- [ ] 2. Implement frontend frame capture module
  - Create FrameCaptureModule class to extract current video frame
  - Implement canvas-based frame capture using drawImage()
  - Add base64 encoding with JPEG compression (quality 75)
  - Handle video not ready and invalid state errors
  - _Requirements: 5.1, 5.2_

- [ ]* 2.1 Write property test for frame capture validation
  - **Property 1: Frame capture produces valid image data**
  - **Validates: Requirements 5.1, 5.2**

- [ ] 3. Create canvas overlay and renderer
  - Create CanvasRenderer class for bounding box visualization
  - Position canvas element over video with matching dimensions
  - Implement canvas resizing on video dimension changes
  - Add CSS for pointer-events passthrough to video controls
  - _Requirements: 6.1, 6.2, 6.4, 6.5_

- [ ]* 3.1 Write property test for canvas dimension matching
  - **Property 2: Canvas dimensions match video display**
  - **Validates: Requirements 6.1, 6.2**

- [ ] 4. Implement bounding box rendering logic
  - Add drawBoundingBox method with coordinate scaling
  - Implement clearCanvas before each render cycle
  - Draw rectangles with class labels and confidence scores
  - Handle multiple overlapping boxes with distinct colors
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 6.3_

- [ ]* 4.1 Write property test for coordinate scaling
  - **Property 3: Bounding box coordinates scale correctly**
  - **Validates: Requirements 3.5**

- [ ]* 4.2 Write property test for canvas clearing
  - **Property 5: Canvas clears before rendering new frame**
  - **Validates: Requirements 6.3**

- [ ]* 4.3 Write property test for bounding box count
  - **Property 8: Bounding box count matches detection count**
  - **Validates: Requirements 3.3**

- [ ] 5. Create detection API client
  - Implement DetectionAPIClient class for backend communication
  - Add async detectFrame method with fetch API
  - Implement error handling with retry logic (max 3 attempts)
  - Parse JSON response and extract detection data
  - _Requirements: 5.3, 5.4, 5.5_

- [ ] 6. Implement VideoDetectionManager controller
  - Create main controller class to orchestrate detection
  - Implement startDetection and stopDetection methods
  - Add detection loop with configurable interval (default 200ms)
  - Integrate frame capture, API client, and canvas renderer
  - Handle detection state management (active/inactive)
  - _Requirements: 1.1, 1.5, 2.1, 2.2, 4.2_

- [ ]* 6.1 Write property test for detection toggle control
  - **Property 6: Detection toggle controls processing**
  - **Validates: Requirements 2.1, 2.2**

- [ ] 7. Add real-time detection UI controls
  - Add "Enable Real-time Detection" toggle button to HTML
  - Update class selection to work with real-time mode
  - Wire toggle button to VideoDetectionManager start/stop methods
  - Update selected classes dynamically when checkboxes change
  - _Requirements: 2.1, 2.2, 2.4_

- [ ] 8. Implement status and feedback indicators
  - Add status display for detection active/inactive state
  - Show processing indicator during API requests
  - Display error messages when detection fails
  - Show detection count for current frame
  - Update status messages with appropriate styling
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [ ] 9. Add video event handlers
  - Handle video pause event to maintain detection results
  - Handle video seek event to trigger new detection
  - Handle video play event to resume detection loop
  - Handle video resize event to update canvas dimensions
  - _Requirements: 1.2, 1.3, 6.2_

- [ ] 10. Implement performance optimizations
  - Add request cancellation for pending detections when new frame ready
  - Implement adaptive frame rate based on backend response time
  - Skip frames if detection takes longer than interval
  - Add frame compression quality adjustment
  - _Requirements: 4.2, 4.3_

- [ ] 11. Integrate real-time detection with existing UI
  - Update index.html to include canvas overlay
  - Add JavaScript modules for detection classes
  - Initialize VideoDetectionManager on page load
  - Maintain compatibility with existing offline detection mode
  - Hide/show appropriate controls based on detection mode
  - _Requirements: 1.1, 2.3_

- [ ] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 13. Add integration tests
  - Test end-to-end flow: upload video → enable detection → verify boxes
  - Test class selection changes during playback
  - Test video seeking with detection active
  - Test error recovery from backend failures
  - _Requirements: All_

- [ ]* 14. Add performance and browser compatibility tests
  - Measure detection latency and frame processing rate
  - Test on Chrome, Firefox, and Safari browsers
  - Verify canvas overlay positioning across browsers
  - Test with different video resolutions and formats
  - _Requirements: 4.1, 4.5_
