"""
SCADA Anomaly Detection API
----------------------------
REST API and WebSocket endpoints for the detection server.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import json
import asyncio
import logging
import os
import time

# Import our modules
import sys

from app.inference.detector import SCADADetector, AlertLevel, get_detector
from app.simulator.data_generator import SCADASimulator, AttackType

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="SCADA Anomaly Detection Server",
    description="Real-time anomaly detection for SCADA/ICS systems using IF + Transformer ensemble",
    version="1.0.0"
)

# Static files and templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global instances
detector: Optional[SCADADetector] = None
simulator: Optional[SCADASimulator] = None
active_websockets: List[WebSocket] = []


# ============================================================================
# Request/Response Models
# ============================================================================

class SensorReading(BaseModel):
    """Single sensor reading."""
    values: List[float]
    timestamp: Optional[float] = None


class BatchReadings(BaseModel):
    """Batch of sensor readings."""
    readings: List[List[float]]


class DetectionResponse(BaseModel):
    """Response from detection endpoint."""
    timestamp: float
    is_anomaly: bool
    alert_level: str
    if_score: float
    transformer_error: float
    ensemble_score: float
    detection_source: str
    message: str
    latency_ms: float


class LoadModelRequest(BaseModel):
    """Request to load models."""
    isolation_forest_path: Optional[str] = None
    transformer_model_path: Optional[str] = None
    transformer_scaler_path: Optional[str] = None


class ThresholdUpdate(BaseModel):
    """Update detection thresholds."""
    if_threshold: Optional[float] = None
    transformer_threshold: Optional[float] = None


class AttackRequest(BaseModel):
    """Request to inject attack."""
    attack_type: str
    duration_samples: int = 50
    target_sensor: int = 0
    intensity: float = 1.0


# ============================================================================
# Startup/Shutdown
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global detector, simulator
    
    detector = get_detector()
    simulator = SCADASimulator(n_features=51)
    
    import os
    
    # Load IF model - ONLY if not already loaded
    if not detector.if_loaded:
        if_path = os.environ.get('SCADA_IF_MODEL')
        if if_path and os.path.exists(if_path):
            detector.load_isolation_forest(if_path)
    
    # Load Transformer - ONLY if not already loaded
    if not detector.transformer_loaded:
        transformer_path = os.environ.get('SCADA_TRANSFORMER_MODEL')
        scaler_path = os.environ.get('SCADA_SCALER')
        if transformer_path and scaler_path:
            if os.path.exists(transformer_path) and os.path.exists(scaler_path):
                detector.load_transformer(transformer_path, scaler_path)
    
    # Set thresholds
    if_thresh = float(os.environ.get('SCADA_IF_THRESHOLD', '-0.05'))
    trans_thresh = float(os.environ.get('SCADA_TRANSFORMER_THRESHOLD', '0.278'))
    detector.set_thresholds(if_thresh, trans_thresh)
    
    logger.info("SCADA Detection Server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global simulator
    if simulator:
        simulator.stop_streaming()
    logger.info("Server shutdown")


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "if_loaded": detector.if_loaded if detector else False,
        "transformer_loaded": detector.transformer_loaded if detector else False
    }


@app.get("/stats")
async def get_stats():
    """Get detector statistics."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    stats = detector.get_stats()
    stats['simulator'] = simulator.get_state() if simulator else None
    return stats


@app.post("/models/load")
async def load_models(request: LoadModelRequest):
    """Load detection models."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    results = {}
    
    if request.isolation_forest_path:
        try:
            detector.load_isolation_forest(request.isolation_forest_path)
            results['isolation_forest'] = "loaded"
        except Exception as e:
            results['isolation_forest'] = f"error: {str(e)}"
    
    if request.transformer_model_path and request.transformer_scaler_path:
        try:
            detector.load_transformer(
                request.transformer_model_path,
                request.transformer_scaler_path
            )
            results['transformer'] = "loaded"
        except Exception as e:
            results['transformer'] = f"error: {str(e)}"
    
    return {"results": results, "stats": detector.get_stats()}


@app.post("/thresholds")
async def update_thresholds(update: ThresholdUpdate):
    """Update detection thresholds."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    detector.set_thresholds(
        if_threshold=update.if_threshold,
        transformer_threshold=update.transformer_threshold
    )
    
    return {
        "if_threshold": detector.if_threshold,
        "transformer_threshold": detector.transformer_threshold
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_single(reading: SensorReading):
    """Run detection on a single sensor reading."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    sample = np.array(reading.values, dtype=np.float32)
    result = detector.detect(sample, reading.timestamp)
    
    return DetectionResponse(
        timestamp=result.timestamp,
        is_anomaly=result.is_anomaly,
        alert_level=result.alert_level.name,
        if_score=result.if_score,
        transformer_error=result.transformer_error,
        ensemble_score=result.ensemble_score,
        detection_source=result.detection_source,
        message=result.message,
        latency_ms=result.latency_ms
    )


@app.post("/detect/batch")
async def detect_batch(batch: BatchReadings):
    """Run detection on batch of readings."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    samples = np.array(batch.readings, dtype=np.float32)
    results = detector.detect_batch(samples)
    
    return {
        "total": len(results),
        "anomalies": sum(1 for r in results if r.is_anomaly),
        "results": [
            {
                "timestamp": r.timestamp,
                "is_anomaly": r.is_anomaly,
                "alert_level": r.alert_level.name,
                "ensemble_score": r.ensemble_score
            }
            for r in results
        ]
    }


@app.post("/reset")
async def reset_detector():
    """Reset detector state."""
    if detector:
        detector.reset()
    return {"status": "reset"}


# ============================================================================
# Simulator Endpoints
# ============================================================================

@app.get("/simulator/state")
async def simulator_state():
    """Get simulator state."""
    if not simulator:
        raise HTTPException(status_code=500, detail="Simulator not initialized")
    return simulator.get_state()


@app.post("/simulator/start")
async def start_simulator(samples_per_second: float = 10, attack_probability: float = 0.02):
    """Start the simulator streaming."""
    loop = asyncio.get_running_loop()
    if not simulator:
        raise HTTPException(status_code=500, detail="Simulator not initialized")
    
    async def broadcast_sample(sample, is_attack, attack_type):
        """Broadcast sample to all WebSocket clients."""
        try:
            # Ensure numpy array
            if not isinstance(sample, np.ndarray):
                sample = np.array(sample, dtype=np.float32)
            
            result = await loop.run_in_executor(
            None, detector.detect, sample
        ) if detector else None
            
            message = {
                "type": "detection",
                "timestamp": time.time(),
                "values": sample[:10].tolist(),
                "is_attack_ground_truth": bool(is_attack),
                "attack_type": str(attack_type),
                "is_anomaly": bool(result.is_anomaly) if result else False,
                "alert_level": str(result.alert_level.name.lower()) if result else "low",
                "if_score": float(result.if_score) if result else 0.0,
                "transformer_score": float(result.transformer_error) if result else 0.0,
                "message": str(result.message) if result else "",
                "latency_ms": float(result.latency_ms) if result else 0.0
            }
            
            disconnected = []
            for ws in active_websockets:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.error(f"WebSocket send error: {e}")
                    disconnected.append(ws)
            
            for ws in disconnected:
                active_websockets.remove(ws)
                
        except Exception as e:
            logger.error(f" broadcast_sample error: {e}")
            import traceback
            traceback.print_exc()
    
    def sync_callback(sample, is_attack, attack_type):
        try:
            asyncio.run_coroutine_threadsafe(
                broadcast_sample(sample, is_attack, attack_type),
                loop
            )
        except Exception as e:
            logger.error(f" sync_callback error: {e}")
    
    simulator.start_streaming(
        callback=sync_callback,
        samples_per_second=samples_per_second,
        attack_probability=attack_probability
    )
    
    return {"status": "started", "rate": samples_per_second}


@app.post("/simulator/stop")
async def stop_simulator():
    """Stop the simulator."""
    if simulator:
        simulator.stop_streaming()
    return {"status": "stopped"}


@app.post("/simulator/attack")
async def inject_attack(request: AttackRequest):
    """Inject an attack into the simulator."""
    if not simulator:
        raise HTTPException(status_code=500, detail="Simulator not initialized")
    
    try:
        attack_type = AttackType(request.attack_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid attack type: {request.attack_type}")
    
    simulator.start_attack(
        attack_type=attack_type,
        duration_samples=request.duration_samples,
        target_sensor=request.target_sensor,
        intensity=request.intensity
    )
    
    return {
        "status": "attack_started",
        "attack_type": request.attack_type,
        "duration": request.duration_samples
    }


@app.post("/simulator/stop_attack")
async def stop_attack():
    """Stop current attack."""
    if simulator:
        simulator.stop_attack()
    return {"status": "attack_stopped"}


@app.post("/simulator/reset")
async def reset_simulator():
    """Reset simulator."""
    if simulator:
        simulator.reset()
    return {"status": "reset"}


# ============================================================================
# WebSocket for Real-Time Updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates."""
    await websocket.accept()
    active_websockets.append(websocket)
    
    logger.info(f"WebSocket connected. Total connections: {len(active_websockets)}")
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "detect":
                # Client sent sensor values for detection
                values = message.get("values", [])
                if values and detector:
                    sample = np.array(values, dtype=np.float32)
                    result = detector.detect(sample)
                    
                    await websocket.send_json({
                        "type": "detection_result",
                        "is_anomaly": result.is_anomaly,
                        "alert_level": result.alert_level.name,
                        "ensemble_score": result.ensemble_score,
                        "message": result.message
                    })
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(active_websockets)}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
