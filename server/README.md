# Depth Anything V2 API Server

This directory contains the complete API service for Depth Anything V2, organized as a self-contained server package.

## 📁 Structure

```
server/
├── app.py                    # FastAPI application
├── models.py                 # Pydantic request/response models
├── depth_service.py          # Core depth prediction service
├── run_api_server.py         # Server startup script
├── test_api_client.py        # Client test script
├── requirements.txt          # API dependencies
├── README_API.md            # Detailed API documentation
├── client/                   # API client implementation
│   ├── depth_client.py       # OpenAI-style client
│   └── __init__.py
└── config/                   # Configuration files
    └── servers.yaml          # Server configurations
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd server
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python run_api_server.py
```

### 3. Test the Client
```bash
python test_api_client.py
```

## 📖 Documentation

See `README_API.md` for complete API documentation, usage examples, and deployment instructions.

## 🔧 Usage

The server provides OpenAI-compatible endpoints for depth prediction. You can use it from anywhere by just installing the client dependencies and pointing to your server URL.

**Example:**
```python
from client.depth_client import predict_depth_simple

result = predict_depth_simple(
    image="path/to/image.jpg",
    camera_intrinsics={'fx': 1000.0, 'fy': 1000.0, 'cx': 640.0, 'cy': 480.0},
    base_url="http://your-server:8000"
)
``` 