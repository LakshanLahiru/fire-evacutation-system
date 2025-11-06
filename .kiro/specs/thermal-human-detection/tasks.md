# Implementation Plan

- [x] 1. Set up thermal detection service foundation


  - Create `services/thermal_detection.py` based on provided ThermalHumanDetector code
  - Adapt the existing code for API service usage (remove main/CLI functionality)
  - Configure model loading to use `AI_models/human thermal detection v1.pt`
  - _Requirements: 4.1, 4.2, 4.5_

- [x] 2. Implement core detection functionality

  - [ ] 2.1 Create image enhancement pipeline
    - Implement histogram equalization, CLAHE, and gamma correction methods
    - Add preprocessing functions for thermal image optimization
    - _Requirements: 1.2_

  
  - [ ] 2.2 Implement YOLO-based human detection
    - Integrate YOLO model for person detection with confidence threshold 0.15
    - Process enhanced images and extract detection results

    - _Requirements: 1.1_
  
  - [ ] 2.3 Add detection merging and filtering
    - Implement IoU calculation for overlapping detections
    - Create detection merging algorithm with 0.5 IoU threshold
    - _Requirements: 1.3_
  
  - [ ]* 2.4 Write unit tests for detection service
    - Test image enhancement methods
    - Test detection merging algorithms
    - Test coordinate conversion functions
    - _Requirements: 1.1, 1.2, 1.3_

- [ ] 3. Create data models and response structures
  - [x] 3.1 Define detection data models


    - Create Detection, DetectionResult, and response models using Pydantic
    - Add validation for input parameters and image data
    - _Requirements: 1.4, 2.3_
  

  - [ ] 3.2 Implement coordinate conversion utilities
    - Add functions to convert detection coordinates to evacuation matrix format
    - Ensure compatibility with existing evacuation system coordinate system
    - _Requirements: 3.1_


- [ ] 4. Extend API endpoints for thermal detection
  - [x] 4.1 Add thermal detection endpoints to api/endpoints.py

    - Create POST /thermal-detection/image endpoint for single image processing
    - Create POST /thermal-detection/batch endpoint for multiple images
    - Add GET /thermal-detection/status endpoint for service health
    - _Requirements: 2.1, 2.4_
  

  - [ ] 4.2 Implement request handling and validation
    - Add file upload handling for image data
    - Implement request validation and error responses
    - Add response formatting with detection results and metadata
    - _Requirements: 2.2, 2.3_

  
  - [ ] 4.3 Add error handling and logging
    - Implement comprehensive error handling for all failure scenarios
    - Add proper HTTP status codes and error messages
    - Include logging for debugging and monitoring
    - _Requirements: 2.4, 4.4_



- [ ] 5. Create standalone thermal detection endpoints
  - [x] 5.1 Add new thermal detection endpoints to api/endpoints.py

    - Create separate endpoints that do not modify existing evacuation functionality
    - Implement endpoints for image upload and detection processing
    - _Requirements: 2.1, 2.2_
  
  - [ ] 5.2 Provide detection results in compatible format
    - Return detection coordinates that could be used with existing evacuation system
    - Include metadata about detected persons and confidence scores
    - _Requirements: 3.1, 3.2_

- [ ]* 6. Add comprehensive testing and validation
  - [ ]* 6.1 Create integration tests
    - Test end-to-end detection and evacuation integration
    - Test API endpoints with various input scenarios
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [ ]* 6.2 Add performance and error handling tests
    - Test processing time requirements (under 2 seconds for single images)

    - Test error scenarios and proper error responses
    - Test batch processing capabilities
    - _Requirements: 2.2, 2.4_



- [ ] 7. Update dependencies and configuration
  - [ ] 7.1 Update requirements.txt
    - Add ultralytics, opencv-python, and other required dependencies for thermal detection
    - Ensure compatibility with existing dependencies
    - _Requirements: 4.1_
  
  - [ ] 7.2 Configure model path for existing AI_models folder
    - Update service to use `AI_models/human thermal detection v1.pt` model path
    - Add model file validation and loading error handling
    - _Requirements: 4.2, 4.5_