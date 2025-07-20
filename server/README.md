# Depth Anything V2 API Server

OpenAI-style API service for depth prediction using Depth Anything V2. Run on your lab server and query from anywhere.

## 🚀 Quick Start

### Server Setup
```bash
pip install -r server/requirements_server.txt
python server/run_api_server.py
```

### Client Usage
```bash
pip install -r server/requirements_client.txt
python server/test_api_client.py
```

## 🔐 Authentication

The API supports OpenAI-style API key authentication using Bearer tokens.

### Server Configuration

**Start server with API key (authentication required):**
```bash
python server/run_api_server.py --api-key "sk-your-api-key"
```

**Start server without authentication (no API key needed):**
```bash
python server/run_api_server.py
```

### Client Usage with Authentication

```python
from server.client.depth_client import predict_pointcloud

# With API key (if server requires authentication)
result = predict_pointcloud(
    image=image_base64,
    camera_intrinsics=cam_int,
    base_url="http://your-server:8000",
    api_key="sk-your-api-key"
)

# Without API key (if server doesn't require authentication)
result = predict_pointcloud(
    image=image_base64,
    camera_intrinsics=cam_int,
    base_url="http://your-server:8000"
)
```

### Configuration File

Update `server/config/servers.yaml` with your API keys:

```yaml
depth-anything-v2-secure:
  base_url: "https://your-secure-server:8000"
  api_key: "sk-your-actual-api-key-here"
  model: "depth-anything-v2"
  description: "Secure server with authentication"
```

## 📁 Structure
```
server/
├── api_server.py             # FastAPI app with 3 endpoints
├── models.py                 # Request/response models
├── depth_service.py          # Core prediction service
├── run_api_server.py         # Server startup
├── test_api_client.py        # Client test script
├── requirements_server.txt   # Server dependencies
├── requirements_client.txt   # Client dependencies
├── client/                   # Client library
│   └── depth_client.py
└── config/                   # Server configs
    └── servers.yaml
```

## 🔧 API Endpoints

### `/pc` - Point Cloud
**POST** - Get 3D point cloud from image

**Headers:**
```
Authorization: Bearer sk-your-api-key  # Required if authentication enabled
Content-Type: application/json
```

**Request Payload:**
```json
{
  "image": "base64_encoded_cv2_bgr_image",
  "camera_intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 480.0
  },
  "max_depth": 1.0,
  "encoder": "vitl",
  "dataset": "hypersim",
  "model_input_size": 518
}
```

**Response:**
```json
{
  "pointcloud": "base64_encoded_float32_pointcloud_array",
  "pointcloud_shape": [307200, 3],
  "depth_map": "base64_encoded_float32_depth_array",
  "depth_shape": [480, 640],
  "min": 0.1,
  "max": 0.95
}
```

### `/metric_depth` - Metric Depth Map
**POST** - Get metric depth map (scaled by max_depth)

**Headers:**
```
Authorization: Bearer sk-your-api-key  # Required if authentication enabled
Content-Type: application/json
```

**Request Payload:**
```json
{
  "image": "base64_encoded_cv2_bgr_image",
  "camera_intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 480.0
  },
  "max_depth": 1.0,
  "encoder": "vitl",
  "dataset": "hypersim",
  "model_input_size": 518
}
```

**Response:**
```json
{
  "depth_map": "base64_encoded_float32_depth_array",
  "shape": [480, 640],
  "min": 0.1,
  "max": 0.95
}
```

### `/rel_depth` - Relative Depth Map
**POST** - Get relative depth map (max_depth=1.0)

**Headers:**
```
Authorization: Bearer sk-your-api-key  # Required if authentication enabled
Content-Type: application/json
```

**Request Payload:**
```json
{
  "image": "base64_encoded_cv2_bgr_image",
  "camera_intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 480.0
  },
  "encoder": "vitl",
  "dataset": "hypersim",
  "model_input_size": 518
}
```

**Response:**
```json
{
  "depth_map": "base64_encoded_float32_depth_array",
  "shape": [480, 640],
  "min": 0.0,
  "max": 1.0
}
```

### `/health` - Health Check
**GET** - Server status

**Headers:** No authentication required

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "gpu_count": 2,
  "models_loaded": ["vitl_hypersim_518_1.0"]
}
```

## 🖥️ Client Usage

### Basic Usage
```python
from server.client.depth_client import encode_image, predict_pointcloud, decode_pointcloud, decode_depth_map

# Encode image
image_base64 = encode_image("path/to/image.jpg")

# Get pointcloud
result = predict_pointcloud(
    image=image_base64,
    camera_intrinsics={'fx': 1000.0, 'fy': 1000.0, 'cx': 640.0, 'cy': 480.0},
    base_url="http://your-server:8000",
    max_depth=1.0,
    api_key="sk-your-api-key"  # Optional
)

# Decode results
pointcloud = decode_pointcloud(result['pointcloud'], result['pointcloud_shape'])
depth_map = decode_depth_map(result['depth_map'])

print(f"Pointcloud shape: {pointcloud.shape}")
print(f"Depth map shape: {depth_map.shape}")
```

### All Functions
```python
from server.client.depth_client import *

# Point cloud (includes both pointcloud and depth)
pc_result = predict_pointcloud(image=image_base64, camera_intrinsics=cam_int, 
                             max_depth=1.0, api_key="sk-your-api-key")

# Metric depth
metric_result = predict_metric_depth(image=image_base64, camera_intrinsics=cam_int, 
                                   max_depth=1.0, api_key="sk-your-api-key")

# Relative depth
rel_result = predict_relative_depth(image=image_base64, camera_intrinsics=cam_int, 
                                  api_key="sk-your-api-key")

# Health check
health = get_health("http://your-server:8000", api_key="sk-your-api-key")
```

## ⚙️ Configuration

### Server Config (`server/config/servers.yaml`)
```yaml
depth-anything-v2-lab:
  base_url: "http://your-lab-server:8000"
  api_key: "sk-your-api-key"  # Set to null for no auth
  model: "depth-anything-v2"
```

## 🚀 Deployment

### On Lab Server (with authentication)
```bash
git clone <your-repo>
cd Depth-Anything-V2
pip install -r server/requirements_server.txt

# Start with authentication
python server/run_api_server.py --host 0.0.0.0 --port 8000 --api-key "sk-your-api-key"
```

### From Any Client
```bash
pip install -r server/requirements_client.txt

# Use in your code
from server.client.depth_client import predict_pointcloud
result = predict_pointcloud(image=image_base64, camera_intrinsics=cam_int, 
                          base_url="http://your-lab-server:8000",
                          api_key="sk-your-api-key")
```

## 🔍 API Documentation
- **Interactive Docs**: http://localhost:8000/docs (auto-generated by FastAPI)
- **Health Check**: http://localhost:8000/health (no authentication required)

**About `/docs` endpoint:**
FastAPI automatically generates interactive API documentation at `/docs` when you run the server. This provides:
- Swagger UI interface to test all endpoints
- Request/response schemas
- Authentication requirements
- Try-it-out functionality
- No additional setup required - it's built into FastAPI

## 📝 Notes
- **No Core Changes**: API wraps existing `DepthAny2` without modifications
- **Model Caching**: Models cached in memory for speed
- **Image Support**: Accepts base64 or URL (like OpenAI)
- **GPU Auto-detect**: Falls back to CPU if CUDA unavailable
- **Multi-GPU Support**: Round-robin distribution across available GPUs (set CUDA_VISIBLE_DEVICES)
- **Authentication**: Automatic - enabled if API key provided, disabled if not
- **Flexible Auth**: Simple on/off based on API key presence
- **Raw Depth Values**: Returns raw float32 depth arrays (no normalization)
- **Enhanced PointCloud**: Includes both pointcloud and depth information

## 🔍 CODE UNDERSTANDING

### Where to Start Reading Code

**1. Entry Point: `run_api_server.py`**
- Start here to understand how the server starts
- Simple script that launches the FastAPI app with CLI arguments
- Sets up host/port and enables auto-reload for development
- **NEW**: Configures API key authentication via command line arguments

**2. API Layer: `api_server.py`**
- FastAPI application with 3 main endpoints (`/pc`, `/metric_depth`, `/rel_depth`)
- Each endpoint validates requests using Pydantic models from `models.py`
- Calls the core service in `depth_service.py` for actual predictions
- Handles HTTP errors and response formatting
- **NEW**: Includes authentication middleware with `verify_api_key()` dependency
- **NEW**: VLLM-style request logging with timestamps and IP addresses
- **NEW**: Async/await patterns for better concurrency

**3. Data Models: `models.py`**
- Pydantic models for request/response validation
- `DepthRequest`: Input payload for all endpoints (image, camera_intrinsics, etc.)
- `PointCloudResponse`: Output for `/pc` endpoint (includes both pointcloud and depth)
- `DepthResponse`: Output for depth map endpoints
- `HealthResponse`: Server status information with GPU count

**4. Core Logic: `depth_service.py`**
- **Main service class**: `DepthService` wraps the original `DepthAny2` model
- **Model caching**: Stores models in memory by encoder/dataset/size/depth combination
- **Multi-GPU support**: Round-robin distribution across available GPUs
- **Image processing**: Handles base64 decoding and URL fetching
- **Three prediction methods**: `predict_pointcloud()`, `predict_metric_depth()`, `predict_relative_depth()`
- **Camera intrinsics**: Converts Pydantic model to numpy array format
- **Raw depth values**: Returns unnormalized float32 depth arrays

**5. Client Library: `client/depth_client.py`**
- Functions to call the API endpoints from Python
- Image encoding/decoding utilities
- Pointcloud and depth map decoding functions (updated for raw float32 format)
- Error handling and response validation
- **NEW**: API key support in all functions with `_get_headers()` helper

**6. Test Script: `test_api_client.py`**
- Demonstrates how to use all endpoints
- Loads camera intrinsics from YAML file
- Tests health, pointcloud, metric depth, and relative depth endpoints
- Visualizes results with matplotlib
- **NEW**: Includes authentication testing with valid/invalid API keys
- **UPDATED**: Fixed shape references for new response formats

### Data Flow

```
Client Request → api_server.py → Authentication → models.py (validation) → depth_service.py → DepthAny2 → Response
```

1. **Request comes in** to FastAPI endpoint in `api_server.py`
2. **Authentication check** verifies API key if enabled
3. **Pydantic validates** the request using models in `models.py`
4. **Service processes** the request in `depth_service.py`:
   - Decodes image (base64 or URL)
   - Gets cached model or loads new one (distributed across GPUs)
   - Runs depth prediction
   - Converts to pointcloud if needed
   - Encodes response as base64 (raw float32 format)
5. **Response returned** through FastAPI with proper HTTP status

### Key Dependencies

- **FastAPI**: Web framework for API endpoints
- **Pydantic**: Data validation and serialization
- **DepthAny2**: Original depth prediction model (from parent directory)
- **OpenCV/PIL**: Image processing
- **NumPy**: Numerical operations
- **PyTorch**: Deep learning framework
- **NEW**: **HTTPBearer**: Authentication middleware
- **NEW**: **asyncio**: Concurrent request handling
- **NEW**: **logging**: VLLM-style request logging

### Extension Points

- **Add authentication**: ✅ **COMPLETED** - API key authentication implemented
- **Add new endpoints**: Add new routes in `api_server.py` and methods in `depth_service.py`
- **Add caching**: Implement Redis/database caching in `depth_service.py`
- **Add logging**: ✅ **COMPLETED** - Structured logging throughout the service
- **Add monitoring**: Add metrics collection and health checks 