# Design Document

## Overview

The thermal human detection feature integrates YOLO-based computer vision capabilities into the existing evacuation system. This design extends the current FastAPI architecture by adding a new service module for thermal detection and corresponding API endpoints. The system will process thermal imagery to detect human presence and provide location data that can be integrated with evacuation routing.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI App    │───▶│   Services      │
│                 │    │                  │    │                 │
│ - Web Interface │    │ - /evacuation    │    │ - visualize.py  │
│ - Mobile App    │    │ - /thermal-det   │    │ - fire_model.py │
│ - Emergency     │    │ - /download      │    │ - thermal_det.py│
│   Console       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   AI Models      │
                       │                  │
                       │ - human2.pt      │
                       │ - YOLO Model     │
                       └──────────────────┘
```

### Service Integration

The thermal detection service will integrate with the existing architecture:

- **API Layer**: New endpoints in `api/endpoints.py` for thermal detection
- **Service Layer**: New `services/thermal_detection.py` module
- **Model Layer**: YOLO model integration for human detection
- **Data Flow**: Thermal detection results can feed into evacuation routing

## Components and Interfaces

### 1. ThermalDetectionService

**Location**: `services/thermal_detection.py`

**Responsibilities**:
- Load and manage YOLO model
- Process thermal imagery with multiple enhancement techniques
- Detect humans and return bounding box coordinates
- Merge overlapping detections
- Convert coordinates to evacuation matrix format

**Key Methods**:
```python
class ThermalDetectionService:
    def __init__(self, model_path: str)
    def detect_humans_in_image(self, image_data: bytes) -> DetectionResult
    def detect_humans_in_video_frame(self, frame: np.ndarray) -> DetectionResult
    def apply_thermal_enhancements(self, frame: np.ndarray) -> List[np.ndarray]
    def merge_detections(self, detections: List[Detection]) -> List[Detection]
    def convert_to_matrix_coordinates(self, detections: List[Detection], matrix_dims: Tuple[int, int]) -> List[Tuple[int, int]]
```

### 2. API Endpoints

**Location**: `api/endpoints.py` (extended)

**New Endpoints**:

1. **POST /thermal-detection/image**
   - Accept single image file
   - Return detection results with person locations
   - Support multiple image formats (PNG, JPG, thermal formats)

2. **POST /thermal-detection/batch**
   - Accept multiple images for batch processing
   - Return aggregated detection results
   - Support processing optimization

3. **GET /thermal-detection/status**
   - Return service health and model information
   - Provide processing statistics

### 3. Detection Models

**DetectionResult**:
```python
@dataclass
class DetectionResult:
    detections: List[Detection]
    processing_time: float
    image_dimensions: Tuple[int, int]
    enhancement_methods_used: List[str]
    timestamp: datetime
```

**Detection**:
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center_point: Tuple[int, int]
    matrix_coordinates: Optional[Tuple[int, int]]
    enhancement_method: int
```

### 4. Integration Points

**With Evacuation System**:
- Detection coordinates can be used as start points for evacuation routing
- Multiple detected persons can generate multiple evacuation routes
- Detection confidence can influence route priority

**With Visualization System**:
- Detected person locations can be overlaid on evacuation maps
- Detection history can show person movement patterns
- Integration with existing visualization in `services/visualize.py`

## Data Models

### Input Data Models

```python
class ThermalImageRequest(BaseModel):
    image_data: str  # Base64 encoded image
    matrix_dimensions: Optional[Tuple[int, int]] = None
    confidence_threshold: float = 0.15
    enhancement_methods: List[str] = ["histogram", "clahe", "gamma"]

class BatchDetectionRequest(BaseModel):
    images: List[ThermalImageRequest]
    merge_results: bool = True
```

### Output Data Models

```python
class DetectionResponse(BaseModel):
    success: bool
    detections: List[Detection]
    processing_time: float
    total_persons_detected: int
    matrix_coordinates: List[Tuple[int, int]]
    metadata: Dict[str, Any]

class BatchDetectionResponse(BaseModel):
    success: bool
    results: List[DetectionResponse]
    summary: Dict[str, Any]
```

## Error Handling

### Error Categories

1. **Model Loading Errors**
   - Missing model file
   - Corrupted model
   - Insufficient memory

2. **Image Processing Errors**
   - Invalid image format
   - Corrupted image data
   - Unsupported image dimensions

3. **Detection Errors**
   - No detections found
   - Processing timeout
   - Enhancement pipeline failure

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
```

### Error Handling Strategy

- Graceful degradation: If one enhancement method fails, continue with others
- Retry mechanism: Automatic retry for transient failures
- Logging: Comprehensive error logging for debugging
- User feedback: Clear error messages for API consumers

## Testing Strategy

### Unit Tests

1. **ThermalDetectionService Tests**
   - Model loading and initialization
   - Image enhancement pipeline
   - Detection merging algorithms
   - Coordinate conversion accuracy

2. **API Endpoint Tests**
   - Request validation
   - Response format verification
   - Error handling scenarios
   - File upload handling

### Integration Tests

1. **End-to-End Detection Flow**
   - Image upload → processing → detection → response
   - Batch processing workflows
   - Integration with evacuation routing

2. **Performance Tests**
   - Processing time benchmarks
   - Memory usage monitoring
   - Concurrent request handling
   - Large image processing

### Test Data

- Sample thermal images with known human locations
- Edge cases: low contrast, multiple persons, occlusions
- Performance test images of various sizes
- Invalid input data for error testing

### Testing Tools

- pytest for unit and integration tests
- Mock objects for YOLO model during testing
- Test image datasets for validation
- Performance profiling tools

## Implementation Considerations

### Performance Optimization

- Model caching to avoid repeated loading
- Image preprocessing optimization
- Batch processing for multiple images
- Asynchronous processing for large requests

### Security

- Input validation for uploaded images
- File size limits to prevent DoS attacks
- Sanitization of file paths and names
- Rate limiting for API endpoints

### Scalability

- Stateless service design for horizontal scaling
- Model loading optimization for container deployment
- Caching strategies for frequently processed images
- Queue-based processing for high-volume scenarios

### Configuration Management

- Environment variables for model paths
- Configurable detection thresholds
- Enhancement method selection
- Processing timeout settings