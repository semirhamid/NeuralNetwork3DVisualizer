import torch.nn as nn
import json
import asyncio

import websockets


class Trainer:
    @staticmethod
    async def train(model, data_loader, criterion, optimizer, websocket, num_epochs=5):
        model_structure = {
            "total_layers": len(list(model.children())),
            "total_params": sum(p.numel() for p in model.parameters()),
            "total_epochs": num_epochs,
            "layer_details": [
                {
                    "layer_name": name,
                    "input_size": list(layer.weight.shape)[1],
                    "output_size": list(layer.weight.shape)[0],
                }
                for name, layer in model.named_children()
                if isinstance(layer, nn.Linear)
            ],
        }

        for epoch in range(num_epochs):
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                training_data = {"epoch": epoch + 1, "layers": [], "model_structure": model_structure}
                for name, param in model.named_parameters():
                    layer_data = {
                        "layer": name,
                        "weights": param.data.tolist(),
                    }
                    if "weight" in name:
                        training_data["layers"].append(layer_data)
                    elif "bias" in name:
                        training_data["layers"][-1]["biases"] = param.data.tolist()

                try:
                    await websocket.send(json.dumps(training_data))
                    await asyncio.sleep(3)
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed: {e}")
                    return
