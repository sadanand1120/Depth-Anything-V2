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

**Start server with API key:**
```bash
python server/run_api_server.py --api-key "sk-your-api-key" --require-auth
```

**Start server without authentication (default):**
```bash
python server/run_api_server.py
```

### Client Usage with Authentication

```python
from client.depth_client import predict_pointcloud

# With API key
result = predict_pointcloud(
    image=image_base64,
    camera_intrinsics=cam_int,
    base_url="http://your-server:8000",
    api_key="sk-your-api-key"
)

# Without API key (if auth is disabled)
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

**Payload:**
```json
{
  "image": "base64_encoded_image",  // or "image_url": "http://..."
  "camera_intrinsics": {
    "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 480.0
  },
  "max_depth": 1.0,  // Required for scaling
  "encoder": "vitl",  // Optional
  "dataset": "hypersim",  // Optional
  "model_input_size": 518  // Optional
}
```

**Response:**
```json
{
  "pointcloud": "base64_encoded_binary",
  "shape": [307200, 3]
}
```

### `/metric_depth` - Metric Depth Map
**POST** - Get metric depth map (scaled by max_depth)

**Headers:** Same as `/pc` (API key required if auth enabled)

**Payload:** Same as `/pc` (max_depth required)

**Response:**
```json
{
  "depth_map": "data:image/png;base64,...",
  "shape": [480, 640],
  "min": 0.0, "max": 1.0
}
```

### `/rel_depth` - Relative Depth Map
**POST** - Get relative depth map (max_depth=1.0)

**Headers:** Same as `/pc` (API key required if auth enabled)

**Payload:** Same as `/pc` (max_depth not needed)

**Response:** Same as `/metric_depth`

### `/health` - Health Check
**GET** - Server status

**Headers:** No authentication required

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "models_loaded": ["vitl_hypersim_518_1.0"]
}
```

## 🖥️ Client Usage

### Basic Usage
```python
from client.depth_client import encode_image, predict_pointcloud, decode_pointcloud

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

# Decode result
pointcloud = decode_pointcloud(result['pointcloud'], result['shape'])
```

### All Functions
```python
from client.depth_client import *

# Point cloud
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
python server/run_api_server.py --host 0.0.0.0 --port 8000 --api-key "sk-your-api-key" --require-auth
```

### From Any Client
```bash
pip install -r server/requirements_client.txt

# Use in your code
from client.depth_client import predict_pointcloud
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
- **Authentication**: Optional API key authentication (OpenAI-style Bearer tokens)
- **Flexible Auth**: Can be enabled/disabled per server instance

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

**3. Data Models: `models.py`**
- Pydantic models for request/response validation
- `DepthRequest`: Input payload for all endpoints (image, camera_intrinsics, etc.)
- `PointCloudResponse`: Output for `/pc` endpoint
- `DepthResponse`: Output for depth map endpoints
- `HealthResponse`: Server status information

**4. Core Logic: `depth_service.py`**
- **Main service class**: `DepthService` wraps the original `DepthAny2` model
- **Model caching**: Stores models in memory by encoder/dataset/size/depth combination
- **Image processing**: Handles base64 decoding and URL fetching
- **Three prediction methods**: `predict_pointcloud()`, `predict_metric_depth()`, `predict_relative_depth()`
- **Camera intrinsics**: Converts Pydantic model to numpy array format

**5. Client Library: `client/depth_client.py`**
- Functions to call the API endpoints from Python
- Image encoding/decoding utilities
- Pointcloud and depth map decoding functions
- Error handling and response validation
- **NEW**: API key support in all functions with `_get_headers()` helper

**6. Test Script: `test_api_client.py`**
- Demonstrates how to use all endpoints
- Loads camera intrinsics from YAML file
- Tests health, pointcloud, metric depth, and relative depth endpoints
- Visualizes results with matplotlib
- **NEW**: Includes authentication testing with valid/invalid API keys

### Data Flow

```
Client Request → app.py → Authentication → models.py (validation) → depth_service.py → DepthAny2 → Response
```

1. **Request comes in** to FastAPI endpoint in `api_server.py`
2. **Authentication check** verifies API key if enabled
3. **Pydantic validates** the request using models in `models.py`
4. **Service processes** the request in `depth_service.py`:
   - Decodes image (base64 or URL)
   - Gets cached model or loads new one
   - Runs depth prediction
   - Converts to pointcloud if needed
   - Encodes response as base64
5. **Response returned** through FastAPI with proper HTTP status

### Key Dependencies

- **FastAPI**: Web framework for API endpoints
- **Pydantic**: Data validation and serialization
- **DepthAny2**: Original depth prediction model (from parent directory)
- **OpenCV/PIL**: Image processing
- **NumPy**: Numerical operations
- **PyTorch**: Deep learning framework
- **NEW**: **HTTPBearer**: Authentication middleware

### Extension Points

- **Add authentication**: ✅ **COMPLETED** - API key authentication implemented
- **Add new endpoints**: Add new routes in `api_server.py` and methods in `depth_service.py`
- **Add caching**: Implement Redis/database caching in `depth_service.py`
- **Add logging**: Add structured logging throughout the service
- **Add monitoring**: Add metrics collection and health checks 