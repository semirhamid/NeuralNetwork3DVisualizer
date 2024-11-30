
# MLP-Visualizer-3D

**MLP-Visualizer-3D** is an interactive 3D visualization tool for exploring the structure and weights of a Multilayer Perceptron (MLP) neural network. Built using Python (for generating and training data) and Three.js (for rendering a 3D graph), this tool provides an engaging way to visualize and interact with the structure and dynamics of neural networks.

## 🌟 Features

- **Flexible Neural Network Configuration**  
  Supports any number of simple linear layers and neurons, enabling dynamic customization.  
  ![Image showcasing flexibility](https://github.com/user-attachments/assets/a785c650-5076-463b-be7f-731319d3bb02)

- **Interactive Environment Control**  
  Adjust the layer width, neuron spacing, and other properties of the visualization for an intuitive experience.  
  ![Image showcasing control features](https://github.com/user-attachments/assets/bac494e4-a902-423d-8e3b-138a56807253)

- **Spacious Display Area**  
  A large rendering space for exploring and interacting with even the most complex network structures.  
  ![Image showcasing display area](https://github.com/user-attachments/assets/a5f4fccf-4a47-4ea0-b335-83b5480e248b)

---

## 🗂️ Project Structure

The repository is organized into **backend** and **frontend** directories to separate concerns:

### **Backend**  
Contains the core logic for defining and training the neural network using PyTorch and for handling server-side communication:  
- **`models/`**  
  - `simple_nn.py`: Implements a simple feedforward neural network using PyTorch.  
    ```python
    import torch
    import torch.nn as nn
    
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 8)
            self.fc2 = nn.Linear(8, 4)
            self.fc3 = nn.Linear(4, 2)
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    ```
- **`services/`**  
  - `model_service.py`: Provides APIs for training and predicting with the neural network.  
  - `trainer.py`: Handles the training logic for the MLP model.  

- **`utils/`**  
  - `data_formatter.py`: Prepares data for training and testing.  

- **`websockets/`**  
  - `client_handler.py`: Manages WebSocket communication between the backend and frontend.  
  - `server.py`: Runs the WebSocket server to enable real-time updates.

### **Frontend**  
Focuses on rendering the 3D visualization and enabling user interactions:  
- **`assets/`**  
  - Houses resources like stylesheets and configuration files.  

- **`components/`**  
  - `Edge.tsx`: Renders connections (edges) between neurons in the network.  
  - `NeuralNetwork.tsx`: The main component for visualizing the neural network structure.  
  - `Node.tsx`: Represents individual neurons in the visualization.  
  - `SideBar.tsx`: Provides controls for interacting with the visualization.  
  - `styles.ts`: Contains styling configurations for the components.  
  - `types.ts`: Defines TypeScript types for the neural network.  

- **`utils/`**  
  - `calculations.ts`: Includes utility functions for calculations related to 3D rendering.  
  - `constants.ts`: Contains reusable constants for the application.  

### Additional Files
- **`vite.config.ts`**: Configuration for Vite, the frontend build tool.  
- **`tailwind.config.js`**: Tailwind CSS configuration for styling.  
- **`requirements.txt`**: Lists Python dependencies for the backend.  
- **`package.json`**: Contains dependencies and scripts for the frontend.  

---

## 🚀 How to Run

### Backend
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the WebSocket server:
   ```bash
   python main.py
   ```

### Frontend
1. Install dependencies:
   ```bash
   yarn install
   ```
2. Start the development server:
   ```bash
   yarn dev
   ```

---

## 💡 Contributing

Feel free to open issues or submit pull requests. All contributions are welcome to enhance the features and usability of **MLP-Visualizer-3D**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
