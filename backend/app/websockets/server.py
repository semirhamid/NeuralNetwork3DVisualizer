import websockets
from .client_handler import handle_client


def start_server():
    print("Starting WebSocket server on ws://localhost:8000")
    return websockets.serve(handle_client, "localhost", 8000)
