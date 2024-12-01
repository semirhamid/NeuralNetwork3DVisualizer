import torch.nn as nn
import json
import asyncio
import websockets


class Trainer:
    @staticmethod
    async def train(model, data_loader, criterion, optimizer, websocket, num_epochs=5):
        # Collect model structure details
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

        # Training loop
        for epoch in range(num_epochs):
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Prepare metadata to send
                training_data = {
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "loss": loss.item(),  # Include the loss value
                    "learning_rate": optimizer.param_groups[0]["lr"],  # Include learning rate
                    "batch_size": inputs.size(0),  # Include batch size
                    "model_structure": model_structure,
                    "layers": [],
                }

                # Add weights, biases, and gradients for each layer
                for name, param in model.named_parameters():
                    if "weight" in name or "bias" in name:
                        layer_data = {
                            "layer": name,
                            "weights": param.data.tolist(),  # Weights
                            "gradients": param.grad.tolist() if param.grad is not None else None,  # Gradients
                        }
                        if "bias" in name:
                            training_data["layers"][-1]["biases"] = param.data.tolist()  # Add biases to last layer
                        else:
                            training_data["layers"].append(layer_data)

                # Send data via WebSocket
                try:
                    await websocket.send(json.dumps(training_data))
                    await asyncio.sleep(3)
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed: {e}")
                    return
