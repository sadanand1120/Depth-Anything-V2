# Depth Anything V2 API Server

OpenAI-style API service for depth prediction using Depth Anything V2.

## Quick Start

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

## Authentication

The API supports OpenAI-style API key authentication. If you start the server with an API key, all endpoints except `/health` require authentication.

### Server Configuration

**Basic usage:**
```bash
# Start server with API key (authentication required)
python server/run_api_server.py --api-key "sk-your-api-key"

# Start server without authentication (no API key needed)
python server/run_api_server.py
```

**Advanced configuration:**
```bash
# Production server with multiple workers
python server/run_api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --api-key "sk-your-api-key" \
  --log-level info
```

**Available options:**
- `--host`: Host to bind (default: 0.0.0.0)
- `--port`: Port to bind (default: 8000)
- `--workers`: Number of worker processes (default: 1)
- `--api-key`: API key for authentication
- `--log-level`: Log level (debug/info/warning/error, default: info)

### Client Usage with Authentication

```python
from server.client.depth_client import predict_pointcloud, predict_relative_depth

# With API key (if server requires authentication)
result = predict_pointcloud(
    image=image_base64,
    camera_intrinsics=cam_int,
    base_url="http://your-server:8000",
    api_key="sk-your-api-key"
)

# Without API key (if server doesn't require authentication)
result = predict_relative_depth(
    image=image_base64,
    base_url="http://your-server:8000"
)
```

### Configuration File

Update `server/client/servers.yaml` with your API keys:

```yaml
dany2:
  base_url: "http://10.0.0.211:8069"
  api_key: "smdepth"
  model: "depth-anything-v2"
  description: "robolidar"
```

## Structure
```
server/
├── api_server.py             # FastAPI app with endpoints
├── models.py                 # Request/response models
├── depth_service.py          # Core prediction service
├── run_api_server.py         # Server startup with config
├── test_api_client.py        # Client test script
├── requirements_server.txt   # Server dependencies
├── requirements_client.txt   # Client dependencies
└── client/                   # Client library
    ├── depth_client.py
    └── servers.yaml          # Server configurations
```

## API Endpoints

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
  "worker_id": 0,
  "gpu_id": 0,
  "gpu_count": 2,
  "models_loaded": ["vitl_hypersim_518"],
  "total_model_instances": 2,
  "request_counter": 15
}
```

## Client Usage

### Basic Usage
```python
from server.client.depth_client import encode_image, predict_pointcloud, decode_pointcloud, decode_depth_map

# Encode image
image_base64 = encode_image("path/to/image.jpg")

# Get pointcloud (requires camera intrinsics)
result = predict_pointcloud(
    image=image_base64,
    camera_intrinsics={'fx': 1000.0, 'fy': 1000.0, 'cx': 640.0, 'cy': 480.0},
    base_url="http://your-server:8000",
    max_depth=1.0,
    api_key="sk-your-api-key"  # Optional
)

# Get relative depth (no camera intrinsics needed)
rel_result = predict_relative_depth(
    image=image_base64,
    base_url="http://your-server:8000",
    api_key="sk-your-api-key"  # Optional
)

# Decode results
pointcloud = decode_pointcloud(result['pointcloud'], result['pointcloud_shape'])
depth_map = decode_depth_map(result['depth_map'], result['depth_shape'])

print(f"Pointcloud shape: {pointcloud.shape}")
print(f"Depth map shape: {depth_map.shape}")
```

### All Functions
```python
from server.client.depth_client import *

# Point cloud (includes both pointcloud and depth) - requires camera intrinsics
pc_result = predict_pointcloud(image=image_base64, camera_intrinsics=cam_int, 
                             max_depth=1.0, api_key="sk-your-api-key")

# Metric depth - no camera intrinsics needed
metric_result = predict_metric_depth(image=image_base64, 
                                   max_depth=1.0, api_key="sk-your-api-key")

# Relative depth - no camera intrinsics needed
rel_result = predict_relative_depth(image=image_base64, 
                                  api_key="sk-your-api-key")

# Health check
health = get_health("http://your-server:8000", api_key="sk-your-api-key")
```

## Deployment

### On Lab Server (with authentication)
```bash
git clone <your-repo>
cd Depth-Anything-V2
pip install -r server/requirements_server.txt

# Start with authentication and multiple workers
python server/run_api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --api-key "sk-your-api-key" \
  --log-level info
```

### From Any Client
```bash
pip install -r server/requirements_client.txt

# Use in your code
from server.client.depth_client import predict_pointcloud, predict_relative_depth

# Pointcloud (requires camera intrinsics)
result = predict_pointcloud(image=image_base64, camera_intrinsics=cam_int, 
                          base_url="http://your-lab-server:8000",
                          api_key="sk-your-api-key")

# Metric depth (no camera intrinsics needed)
metric_result = predict_metric_depth(image=image_base64,
                                   base_url="http://your-lab-server:8000",
                                   api_key="sk-your-api-key")

# Relative depth (no camera intrinsics needed)
rel_result = predict_relative_depth(image=image_base64,
                                  base_url="http://your-lab-server:8000",
                                  api_key="sk-your-api-key")
```

## API Documentation
- **Interactive Docs**: http://localhost:8000/docs (auto-generated by FastAPI)
- **Health Check**: http://localhost:8000/health (no authentication required)

## Notes
- **No Core Changes**: API wraps existing `DepthAny2` without modifications
- **Model Caching**: Models cached in memory for speed
- **Image Support**: Accepts base64 or URL (like OpenAI)
- **GPU Auto-detect**: Falls back to CPU if CUDA unavailable
- **Multi-GPU Support**: Request-level round-robin across all GPUs
- **Authentication**: Automatic - enabled if API key provided, disabled if not
- **Raw Depth Values**: Returns raw float32 depth arrays (no normalization)
- **Enhanced PointCloud**: Includes both pointcloud and depth information
- **Camera Intrinsics**: Only required for `/pc` endpoint, not needed for `/metric_depth` or `/rel_depth`