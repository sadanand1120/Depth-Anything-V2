# Depth Anything V2 API Service

This repository now includes an OpenAI-compatible API service for depth prediction using Depth Anything V2. The service allows you to run depth prediction on a server and query it from anywhere without needing to set up the entire repository locally.

## 🚀 Quick Start

### 1. Install API Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
# Start server on default port 8000
python run_api_server.py

# Or with custom host/port
python run_api_server.py --host 0.0.0.0 --port 8000

# For development with auto-reload
python run_api_server.py --reload
```

### 3. Test the API

```bash
# Test the client
python test_api_client.py
```

## 📁 Project Structure

```
Depth-Anything-V2/
├── server/                    # API server implementation
│   ├── app.py                # FastAPI application
│   ├── models.py             # Pydantic request/response models
│   ├── depth_service.py      # Core depth prediction service
│   └── __init__.py
├── client/                   # API client implementation
│   ├── depth_client.py       # OpenAI-style client
│   └── __init__.py
├── config/                   # Configuration files
│   └── servers.yaml          # Server configurations
├── run_api_server.py         # Server startup script
├── test_api_client.py        # Client test script
├── requirements_api.txt      # API-specific requirements
└── README_API.md            # This file
```

## 🔧 API Endpoints

### Main Endpoint: `/v1/depth/predict`

**Method**: POST  
**Format**: OpenAI-compatible chat completion

**Request Example**:
```json
{
  "model": "depth-anything-v2",
  "messages": [
    {
      "role": "user",
      "content": {
        "image": "base64_encoded_image",
        "camera_intrinsics": {
          "fx": 1000.0,
          "fy": 1000.0,
          "cx": 640.0,
          "cy": 480.0
        },
        "options": {
          "encoder": "vitl",
          "dataset": "hypersim",
          "max_depth": 1.0,
          "model_input_size": 518
        }
      }
    }
  ],
  "response_format": "all"
}
```

**Response Example**:
```json
{
  "id": "uuid",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "depth-anything-v2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": {
          "depth_relative": {
            "data": "data:image/png;base64,...",
            "shape": [480, 640],
            "min": 0.0,
            "max": 1.0
          },
          "pointcloud": {
            "data": "base64_encoded_binary",
            "shape": [307200, 3],
            "format": "float32_binary"
          }
        }
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "total_tokens": 2
  }
}
```

### Simple Endpoint: `/v1/depth/predict/simple`

**Method**: POST  
**Format**: Simplified JSON

**Request Example**:
```json
{
  "image": "base64_encoded_image",
  "fx": 1000.0,
  "fy": 1000.0,
  "cx": 640.0,
  "cy": 480.0,
  "encoder": "vitl",
  "dataset": "hypersim",
  "max_depth": 1.0,
  "model_input_size": 518,
  "format": "all"
}
```

### Health Check: `/v1/depth/health`

**Method**: GET  
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "cuda_available": true,
  "loaded_models": ["vitl_hypersim_518_1.0"]
}
```

### Models List: `/v1/depth/models`

**Method**: GET  
**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "depth-anything-v2-vitl-hypersim",
      "object": "model",
      "created": 1234567890,
      "owned_by": "depth-anything-v2"
    }
  ]
}
```

## 🖥️ Client Usage

### OpenAI-Style Client

```python
from client.depth_client import depth_chat_completion, create_depth_request, decode_depth_response

# Create request
messages = create_depth_request(
    image="path/to/image.jpg",
    camera_intrinsics={
        'fx': 1000.0,
        'fy': 1000.0,
        'cx': 640.0,
        'cy': 480.0
    },
    encoder="vitl",
    dataset="hypersim",
    response_format="all"
)

# Send request
response = depth_chat_completion(
    model="depth-anything-v2",
    messages=messages,
    base_url="http://localhost:8000",
    response_format="all"
)

# Decode response
result = decode_depth_response(response)
depth_map = result['depth_relative']
pointcloud = result['pointcloud']
```

### Simple Client

```python
from client.depth_client import predict_depth_simple

result = predict_depth_simple(
    image="path/to/image.jpg",
    camera_intrinsics={
        'fx': 1000.0,
        'fy': 1000.0,
        'cx': 640.0,
        'cy': 480.0
    },
    base_url="http://localhost:8000",
    encoder="vitl",
    dataset="hypersim",
    format="all"
)

if result['success']:
    depth_map = result['depth_relative']
    pointcloud = result['pointcloud']
```

## ⚙️ Configuration

### Server Configuration (`config/servers.yaml`)

```yaml
# Local development server
depth-anything-v2-local:
  base_url: "http://localhost:8000"
  api_key: null
  model: "depth-anything-v2"
  description: "Local development server"

# Lab server (replace with your actual server)
depth-anything-v2-lab:
  base_url: "http://your-lab-server:8000"
  api_key: null
  model: "depth-anything-v2"
  description: "AMRL Lab server with H-100/A-100 GPUs"
```

## 🚀 Deployment

### 1. On Your Lab Server

```bash
# Clone the repository
git clone <your-repo-url>
cd Depth-Anything-V2

# Install dependencies
pip install -r requirements_api.txt

# Start the server
python run_api_server.py --host 0.0.0.0 --port 8000
```

### 2. Update Client Configuration

Edit `config/servers.yaml` with your server details:

```yaml
depth-anything-v2-lab:
  base_url: "http://your-lab-server-ip:8000"
  api_key: null
  model: "depth-anything-v2"
  description: "AMRL Lab server"
```

### 3. Use from Anywhere

```python
# From any machine, just install the client
pip install openai requests pillow numpy

# Use the client
from client.depth_client import predict_depth_simple

result = predict_depth_simple(
    image="path/to/image.jpg",
    camera_intrinsics={'fx': 1000.0, 'fy': 1000.0, 'cx': 640.0, 'cy': 480.0},
    base_url="http://your-lab-server-ip:8000"
)
```

## 🔍 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/v1/depth/health

## 🧪 Testing

### Run Tests

```bash
# Test the API client
python test_api_client.py

# Test server health
curl http://localhost:8000/v1/depth/health

# Test models list
curl http://localhost:8000/v1/depth/models
```

### Compare with Original

The test script includes a comparison function that runs both the original `minimal_pts.py` and the API version to ensure they produce identical results.

## 🔧 Advanced Configuration

### Environment Variables

```bash
export DEPTH_API_HOST=0.0.0.0
export DEPTH_API_PORT=8000
export DEPTH_API_WORKERS=4
```

### Production Deployment

For production, consider using:
- **Gunicorn**: `gunicorn server.app:app -w 4 -k uvicorn.workers.UvicornWorker`
- **Docker**: Create a Dockerfile for containerized deployment
- **Reverse Proxy**: Use nginx for load balancing and SSL termination

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct directory and Python path is set
2. **CUDA Issues**: The service automatically falls back to CPU if CUDA is not available
3. **Memory Issues**: Reduce `model_input_size` or use a smaller encoder (vits instead of vitl)
4. **Port Conflicts**: Change the port using `--port` argument

### Debug Mode

```bash
# Run with debug logging
python run_api_server.py --reload

# Check server logs for detailed error messages
```

## 📝 Notes

- **No Changes to Core Logic**: The API service wraps the existing `DepthAny2` class without modifying it
- **Model Caching**: Models are cached in memory for faster subsequent requests
- **Base64 Encoding**: Images are transmitted as base64-encoded strings
- **OpenAI Compatibility**: The main endpoint follows OpenAI's chat completion format
- **Multiple Formats**: Support for pointcloud, depth maps, or both in responses

## 🤝 Contributing

To extend the API:
1. Add new endpoints in `server/app.py`
2. Update models in `server/models.py`
3. Add corresponding client functions in `client/depth_client.py`
4. Update tests in `test_api_client.py` 