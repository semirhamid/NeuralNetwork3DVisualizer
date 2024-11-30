import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Sidebar, SliderControls } from './SideBar';
import { LayerData, ModelStructure, TrainingData } from './types';
import Node from './Node';
import Edge from './Edge';
import {
  DEFAULT_LAYER_SPACING,
  DEFAULT_NEURON_SPACING,
  DEFAULT_NODE_SIZE,
  LAYER_COLORS,
} from '../constants';
import { calculateNeuronPositions } from '../utils/calculations';

const NeuralNetwork: React.FC = () => {
  const [mlpData, setMlpData] = useState<LayerData[] | null>(null);
  const [modelMetadata, setModelMetadata] = useState<ModelStructure | null>(
    null
  );
  const [layerSpacing, setLayerSpacing] = useState(DEFAULT_LAYER_SPACING);
  const [neuronSpacing, setNeuronSpacing] = useState(DEFAULT_NEURON_SPACING);
  const [nodeSize, setNodeSize] = useState(DEFAULT_NODE_SIZE);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000');

    socket.onopen = () => console.log('WebSocket connection established');
    socket.onmessage = (event) => {
      const receivedData: TrainingData = JSON.parse(event.data);
      console.log('Received data:', receivedData);

      if (receivedData.epoch && receivedData.layers) {
        setMlpData(receivedData.layers);
        setModelMetadata(receivedData.model_structure);
      } else {
        console.error('Invalid data format received from WebSocket');
      }
    };

    socket.onclose = () => console.log('WebSocket connection closed');
    socket.onerror = (error) => console.error('WebSocket error:', error);

    return () => {
      socket.close();
    };
  }, []);

  if (!mlpData || !modelMetadata)
    return <div>Loading Neural Network Visualization...</div>;

  const neuronPositions = calculateNeuronPositions(
    mlpData,
    layerSpacing,
    neuronSpacing
  );

  return (
    <div className="flex overflow-y-hidden max-h-screen">
      {/* Sidebar */}
      <section className="">
        <Sidebar modelMetadata={modelMetadata} />

        {/* Controls */}
        <SliderControls
          layerSpacing={layerSpacing}
          setLayerSpacing={setLayerSpacing}
          neuronSpacing={neuronSpacing}
          setNeuronSpacing={setNeuronSpacing}
          nodeSize={nodeSize}
          setNodeSize={setNodeSize}
        />
      </section>

      {/* Neural Network Visualization */}
      <div className="flex-1 max-h-screen max-w-full w-full">
        <Canvas camera={{ fov: 50, position: [0, 0, -50] }}>
          <OrbitControls />
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          {/* Render Neurons */}
          {neuronPositions.map((layer, layerIndex) =>
            layer.map((position, neuronIndex) => (
              <Node
                key={`neuron-${layerIndex}-${neuronIndex}`}
                position={position}
                color={LAYER_COLORS[layerIndex % LAYER_COLORS.length]}
                nodeSize={nodeSize}
              />
            ))
          )}

          {/* Render Connections */}
          {mlpData.map((layer, layerIndex) => {
            if (layerIndex === 0) return null;

            return layer.weights.map((neuronWeights, targetIndex) =>
              neuronWeights.map((weight, sourceIndex) => {
                const sourcePosition =
                  neuronPositions[layerIndex - 1][sourceIndex];
                const targetPosition = neuronPositions[layerIndex][targetIndex];

                return (
                  <Edge
                    key={`line-${layerIndex}-${sourceIndex}-${targetIndex}`}
                    sourcePosition={sourcePosition}
                    targetPosition={targetPosition}
                    weight={weight}
                  />
                );
              })
            );
          })}
        </Canvas>
      </div>
    </div>
  );
};

export default NeuralNetwork;
