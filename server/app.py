import time
import uuid
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    DepthRequest, DepthResponse, HealthResponse, ModelsResponse,
    ResponseFormat, UserContent, DepthOptions
)
from .depth_service import depth_service

app = FastAPI(
    title="Depth Anything V2 API",
    description="OpenAI-compatible API for depth prediction using Depth Anything V2",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/depth/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    status = depth_service.get_health_status()
    return HealthResponse(**status)

@app.get("/v1/depth/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models_data = depth_service.get_available_models()
    return ModelsResponse(data=models_data['models'])

@app.post("/v1/depth/predict", response_model=DepthResponse)
async def predict_depth(request: DepthRequest):
    """Main depth prediction endpoint with OpenAI-compatible format"""
    
    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    # Find user message with image content
    user_message = None
    for msg in request.messages:
        if msg.role == "user" and isinstance(msg.content, UserContent):
            user_message = msg
            break
    
    if not user_message:
        raise HTTPException(
            status_code=400, 
            detail="No user message with image content found"
        )
    
    content = user_message.content
    
    # Extract options
    options = content.options or DepthOptions()
    
    try:
        # Perform depth prediction
        result = depth_service.predict(
            image_base64=content.image,
            camera_intrinsics={
                'fx': content.camera_intrinsics.fx,
                'fy': content.camera_intrinsics.fy,
                'cx': content.camera_intrinsics.cx,
                'cy': content.camera_intrinsics.cy
            },
            encoder=options.encoder,
            dataset=options.dataset,
            model_input_size=options.model_input_size,
            max_depth=options.max_depth
        )
        
        # Prepare response based on format
        response_content = {}
        
        if request.response_format in [ResponseFormat.DEPTH_RELATIVE, ResponseFormat.ALL]:
            # Encode depth map as base64
            depth_normalized = ((result['depth_relative'] - result['depth_relative'].min()) / 
                              (result['depth_relative'].max() - result['depth_relative'].min()) * 255).astype(np.uint8)
            _, depth_buffer = cv2.imencode('.png', depth_normalized)
            depth_base64 = base64.b64encode(depth_buffer).decode('utf-8')
            response_content['depth_relative'] = {
                'data': f"data:image/png;base64,{depth_base64}",
                'shape': result['depth_relative'].shape,
                'min': float(result['depth_relative'].min()),
                'max': float(result['depth_relative'].max())
            }
        
        if request.response_format in [ResponseFormat.POINTCLOUD, ResponseFormat.ALL]:
            # Encode point cloud as base64 (binary format)
            points_flat = result['pointcloud'].astype(np.float32)
            points_base64 = base64.b64encode(points_flat.tobytes()).decode('utf-8')
            response_content['pointcloud'] = {
                'data': points_base64,
                'shape': result['pointcloud'].shape,
                'format': 'float32_binary'
            }
        
        # Create OpenAI-compatible response
        response_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        choices = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "finish_reason": "stop"
        }]
        
        usage = {
            "prompt_tokens": 1,  # Not applicable for depth prediction
            "completion_tokens": 1,
            "total_tokens": 2
        }
        
        return DepthResponse(
            id=response_id,
            created=timestamp,
            model=request.model,
            choices=choices,
            usage=usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Depth prediction failed: {str(e)}")

@app.post("/v1/depth/predict/simple")
async def predict_depth_simple(
    image: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    encoder: str = "vitl",
    dataset: str = "hypersim",
    max_depth: float = 1.0,
    model_input_size: int = 518,
    format: str = "all"
):
    """Simplified endpoint for direct depth prediction"""
    
    try:
        result = depth_service.predict(
            image_base64=image,
            camera_intrinsics={'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
            encoder=encoder,
            dataset=dataset,
            model_input_size=model_input_size,
            max_depth=max_depth
        )
        
        response = {
            'success': True,
            'image_shape': result['image_shape'],
            'pointcloud_shape': result['pointcloud_shape']
        }
        
        if format in ['depth_relative', 'all']:
            depth_normalized = ((result['depth_relative'] - result['depth_relative'].min()) / 
                              (result['depth_relative'].max() - result['depth_relative'].min()) * 255).astype(np.uint8)
            _, depth_buffer = cv2.imencode('.png', depth_normalized)
            depth_base64 = base64.b64encode(depth_buffer).decode('utf-8')
            response['depth_relative'] = f"data:image/png;base64,{depth_base64}"
        
        if format in ['pointcloud', 'all']:
            points_flat = result['pointcloud'].astype(np.float32)
            points_base64 = base64.b64encode(points_flat.tobytes()).decode('utf-8')
            response['pointcloud'] = points_base64
        
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 