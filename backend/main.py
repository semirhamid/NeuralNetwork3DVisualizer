import asyncio
from websockets import serve
from app.sockets.websocket_handler import websocket_handler

HOST = "localhost"
PORT = 8000

async def main():
    """
    Starts the WebSocket server and runs it indefinitely.
    """
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    try:
        async with serve(websocket_handler, HOST, PORT):
            print("WebSocket server is running...")
            await asyncio.Future()  # Keep the server running forever
    except Exception as e:
        print(f"Error occurred while running the WebSocket server: {e}")
    finally:
        print("WebSocket server shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("WebSocket server stopped manually.")