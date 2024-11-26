import torch
import json
import asyncio
from websockets import WebSocketServerProtocol
from app.models.simple_nn import SimpleNN

async def train_model(websocket: WebSocketServerProtocol):
    model = SimpleNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Dummy training data
    inputs = torch.rand(10, 3)
    targets = torch.rand(10, 2)

    for epoch in range(50):  # Simulate training for 50 epochs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Send weights and biases
        params = {
            "neurons": [
                {"id": i, "layer": "hidden", "weight": param.detach().numpy().tolist()}
                for i, param in enumerate(model.parameters())
            ],
            "weights": [
                {
                    "source": i,
                    "target": i + 1,
                    "value": param.detach().numpy().tolist(),
                    "color": f"#{(i * 123456) % 0xFFFFFF:06x}",
                }
                for i, param in enumerate(model.parameters())
            ],
        }

        await websocket.send(json.dumps(params))
        await asyncio.sleep(0.5)  # Simulate delay

async def websocket_handler(websocket: WebSocketServerProtocol):
    print("Client connected")
    try:
        await train_model(websocket)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Client disconnected")