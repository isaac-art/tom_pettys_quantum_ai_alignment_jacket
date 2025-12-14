from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi import Request
from pydantic import BaseModel
import asyncio
import json
import os
from typing import List, Dict, Optional
from generator_service import Generator, GeneratorConfig, EOT, START_HEADER, END_HEADER, BEGIN_TEXT
from audio_controller import AudioController

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") if os.path.exists("templates") else None

# Initialize generator and global state
print("Initializing generator...")
generator = Generator()
print("Generator initialized and ready!")

# Initialize audio controller
print("Initializing audio controller...")
audio_controller = AudioController()
audio_controller.start()
print("Audio controller initialized and started!")

current_x = 0.0
current_y = 0.0
streaming_active = False

def get_available_vectors() -> List[str]:
    """Get list of available vector names without extensions"""
    vec_dir = "vec"
    if not os.path.exists(vec_dir): return []
    return [os.path.splitext(f)[0] for f in os.listdir(vec_dir) if f.endswith('.gguf')]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Serve index.html directly if templates directory doesn't exist
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("<h1>index.html not found</h1>")

@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})

@app.get("/ball", response_class=HTMLResponse)
async def ball(request: Request):
    return templates.TemplateResponse("ball.html", {"request": request})

@app.get("/hands", response_class=HTMLResponse)
async def hands(request: Request):
    return templates.TemplateResponse("hands.html", {"request": request})

@app.get("/vectors")
async def list_vectors():
    """Return list of available vector names"""
    vectors = get_available_vectors()
    return JSONResponse({"vectors": vectors})

@app.post("/vectors/change")
async def change_vectors(vectors: Dict[str, str]):
    """Change the active control vectors and reset generator"""
    global generator
    x_vector = vectors.get("x", "TPPAST")
    y_vector = vectors.get("y", "TPPRESENT")
    
    try:
        generator.load_vectors(x_vector, y_vector)
        # Reset to original start messages after loading new vectors
        generator.reset_to_original()
        print(f"🔄 Changed vectors to {x_vector} / {y_vector} and reset generator")
        return {"status": "success", "x_vector": x_vector, "y_vector": y_vector}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class BLEPosition(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None

@app.post("/ble/position")
async def update_ble_position(position: BLEPosition):
    """Receive BLE position updates"""
    global current_x, current_y
    if position.x is not None:
        current_x = max(-1.0, min(1.0, position.x))
    if position.y is not None:
        current_y = max(-1.0, min(1.0, position.y))
    
    # Update audio playback speeds based on x/y vectors
    audio_controller.update_speeds(current_x, current_y)
    
    return {"status": "ok", "x": current_x, "y": current_y}

@app.post("/reset")
async def reset_generator():
    """Reset the generator to original start messages"""
    global generator
    generator.reset_to_original()
    print("🔄 Generator reset via API")
    return {"status": "success", "message": "Generator reset"}

@app.websocket("/ws/position")
async def position_endpoint(websocket: WebSocket):
    global current_x, current_y
    await websocket.accept()
    
    try:
        while True:
            # Receive position
            data = await websocket.receive_json()
            current_x = data.get("x", 0)  # -1 to 1
            current_y = data.get("y", 0)   # -1 to 1
            
            # Update audio playback speeds based on x/y vectors
            audio_controller.update_speeds(current_x, current_y)
            
    except Exception as e:
        print(f"Position WebSocket error: {e}")

@app.websocket("/ws/reset")
async def reset_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "reset":
                print("Resetting conversation")
                prompt = data.get("prompt", "Who are you?")
                generator.reset_conversation(prompt)
                await websocket.send_json({
                    "status": "success",
                    "message": "Conversation reset"
                })
            elif data.get("type") == "message":
                print("Adding new user message")
                message = data.get("message", "").strip()
                if message:
                    generator.add_message(message, "user")
                    await websocket.send_json({
                        "status": "success",
                        "message": "Message added"
                    })
    except Exception as e:
        print(f"Reset WebSocket error: {e}")

@app.websocket("/ws/config")
async def config_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "set_allow_eof":
                allow_eof = data.get("allow_eof", True)
                generator.set_allow_eof(allow_eof)
                await websocket.send_json({
                    "status": "success",
                    "message": f"EOF {'enabled' if allow_eof else 'disabled'}"
                })
            elif data.get("type") == "get_history":
                await websocket.send_json({
                    "status": "success",
                    "history": generator.message_history
                })
    except Exception as e:
        print(f"Config WebSocket error: {e}")

@app.websocket("/ws/text")
async def text_endpoint(websocket: WebSocket):
    global current_x, current_y
    await websocket.accept()
    
    try:
        while True:
            # Wait for start signal
            data = await websocket.receive_json()
            if data.get("type") == "start":
                # Generate until EOT
                while True:
                    # Update generator with current position
                    generator.update_controls(current_x, current_y)
                    
                    # Generate next token
                    token = generator.next()
                    
                    # Send token data
                    await websocket.send_json({
                        "content": token.content,
                        "x_strength": token.x_strength,
                        "y_strength": token.y_strength,
                        "is_eot": EOT in token.content
                    })
                    
                    # Stop if we hit EOT
                    if EOT in token.content:
                        break
                    
                    # Small delay to control generation rate
                    await asyncio.sleep(0.05)
                
                # Send completion signal and wait for next user input
                await websocket.send_json({
                    "type": "complete"
                })
                # Important: The outer loop will now wait for the next "start" signal
            
    except Exception as e:
        print(f"Text WebSocket error: {e}")

@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """Continuous token streaming endpoint for index.html"""
    global current_x, current_y, streaming_active
    await websocket.accept()
    streaming_active = True
    print("📡 Stream WebSocket connected - starting token generation...")
    token_count = 0
    
    try:
        # Start generating immediately and continuously
        while True:
            try:
                # Update generator with current BLE position (defaults to 0,0 if no BLE input)
                generator.update_controls(current_x, current_y)
                
                # Update audio playback speeds based on x/y vectors
                audio_controller.update_speeds(current_x, current_y)
                
                # Generate next token
                token = generator.next()
                token_count += 1
                
                # Check if this token contains EOT that triggers reset
                is_reset_token = EOT in token.content
                
                # Format for index.html (expects 'text' and 'vectors' array)
                await websocket.send_json({
                    "text": token.content,
                    "vectors": [
                        {"name": "TPPAST", "strength": token.x_strength},
                        {"name": "TPPRESENT", "strength": token.y_strength}
                    ],
                    "reset": is_reset_token  # Signal to frontend to clear text
                })
                
                # Log first few tokens to confirm it's working
                if token_count <= 5:
                    print(f"  Generated token #{token_count}: {repr(token.content)} (x={token.x_strength:.2f}, y={token.y_strength:.2f})")
                
                # Small delay to control generation rate
                await asyncio.sleep(0.05)
                
                # If we hit a reset token, small pause before continuing
                if is_reset_token:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"⚠️  Error generating token: {e}")
                import traceback
                traceback.print_exc()
                # Continue generating even if one token fails
                await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"❌ Stream WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        streaming_active = False
        print(f"📡 Stream WebSocket disconnected (generated {token_count} tokens)")

if __name__ == "__main__":
    import uvicorn
    import atexit
    
    # Print available vectors on startup
    vectors = get_available_vectors()
    print(f"Available control vectors: {vectors}")
    
    # Cleanup audio on exit
    def cleanup():
        audio_controller.cleanup()
    
    atexit.register(cleanup)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8087)
    finally:
        cleanup()
