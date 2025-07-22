import asyncio
import logging
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from depthany2.server.models import DepthRequest, PointCloudResponse, DepthResponse, HealthResponse
from depthany2.server.depth_service import DepthService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Depth Anything V2 API", version="1.0.0")
security = HTTPBearer(auto_error=False)

API_KEY = "smdepth"
MAX_CONCURRENT_REQUESTS = 4
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
depth_service = DepthService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"{client_ip} - \"{request.method} {request.url.path}\" - Processing")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{client_ip} - \"{request.method} {request.url.path}\" {response.status_code} - {process_time:.3f}s")
    
    return response

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@app.get("/health", response_model=HealthResponse)
async def health():
    return depth_service.get_health()

@app.post("/pc", response_model=PointCloudResponse)
async def get_pointcloud(depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not depth_request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for pointcloud")
    
    if not depth_request.camera_intrinsics:
        raise HTTPException(status_code=400, detail="camera_intrinsics is required for pointcloud")
    
    try:
        async with request_semaphore:
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                depth_service.predict_pointcloud,
                depth_request.image,
                depth_request.image_url,
                depth_request.camera_intrinsics,
                depth_request.encoder,
                depth_request.dataset,
                depth_request.model_input_size,
                depth_request.max_depth
            )
        return PointCloudResponse(**result)
    except Exception as e:
        logger.error(f"Error in pointcloud endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/metric_depth", response_model=DepthResponse)
async def get_metric_depth(depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not depth_request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for metric depth")
    
    try:
        async with request_semaphore:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                depth_service.predict_metric_depth,
                depth_request.image,
                depth_request.image_url,
                depth_request.encoder,
                depth_request.dataset,
                depth_request.model_input_size,
                depth_request.max_depth
            )
        return DepthResponse(**result)
    except Exception as e:
        logger.error(f"Error in metric_depth endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/rel_depth", response_model=DepthResponse)
async def get_relative_depth(depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    try:
        async with request_semaphore:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                depth_service.predict_relative_depth,
                depth_request.image,
                depth_request.image_url,
                depth_request.encoder,
                depth_request.dataset,
                depth_request.model_input_size
            )
        return DepthResponse(**result)
    except Exception as e:
        logger.error(f"Error in relative_depth endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")