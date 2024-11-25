import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls, Text } from '@react-three/drei';
import Node from './Node';

type Neuron = {
  id: number;
  layer: 'input' | 'hidden' | 'output';
  position: [number, number, number];
  weight: number; // Determines node size
};

type Weight = {
  source: number;
  target: number;
  value: number;
  color: string;
};

type MLPData = {
  neurons: Neuron[];
  weights: Weight[];
};

const NeuralNetwork: React.FC = () => {
  const [mlpData, setMlpData] = useState<MLPData | null>(null);

  useEffect(() => {
    // Initialize the MLP data
    const initialData: MLPData = {
      neurons: [
        { id: 1, layer: 'input', position: [-5, 2, 0], weight: 1.8 },
        { id: 2, layer: 'input', position: [-5, 0, 0], weight: 2.0 },
        { id: 3, layer: 'input', position: [-5, -2, 0], weight: 1.6 },
        { id: 4, layer: 'hidden', position: [-2.5, 2, 0], weight: 4 },
        { id: 5, layer: 'hidden', position: [-2.5, 0, 0], weight: 1.7 },
        { id: 6, layer: 'hidden', position: [-2.5, -2, 0], weight: 2.2 },
        { id: 7, layer: 'hidden', position: [0, 2, 0], weight: 3.5 },
        { id: 8, layer: 'hidden', position: [0, 0, 0], weight: 2.8 },
        { id: 9, layer: 'hidden', position: [0, -2, 0], weight: 1.9 },
        { id: 10, layer: 'output', position: [3, 1, 0], weight: 2.2 },
        { id: 11, layer: 'output', position: [3, -1, 0], weight: 2.2 },
      ],
      weights: [
        { source: 1, target: 4, value: Math.random(), color: 'red' },
        { source: 2, target: 5, value: Math.random(), color: 'green' },
        { source: 3, target: 6, value: Math.random(), color: 'blue' },
        { source: 4, target: 7, value: Math.random(), color: 'yellow' },
        { source: 5, target: 8, value: Math.random(), color: 'purple' },
        { source: 6, target: 9, value: Math.random(), color: 'orange' },
        { source: 7, target: 10, value: Math.random(), color: 'pink' },
        { source: 8, target: 11, value: Math.random(), color: 'cyan' },
        { source: 9, target: 10, value: Math.random(), color: 'magenta' },
      ],
    };
    setMlpData(initialData);

    // Update the MLP data every 200 ms
    const interval = setInterval(() => {
      setMlpData(prevData => {
        if (!prevData) return null;

        // Update weights and neuron sizes
        const updatedNeurons = prevData.neurons.map(neuron => ({
          ...neuron,
          weight: Math.random() * 3 + 1, // Random weight between 1 and 4
        }));

        const updatedWeights = prevData.weights.map(weight => ({
          ...weight,
          value: Math.random(), // Random weight value
          target: prevData.neurons[Math.floor(Math.random() * prevData.neurons.length)].id, // Random target neuron
          color: `#${Math.floor(Math.random() * 16777215).toString(16)}`, // Random color
        }));

        return {
          neurons: updatedNeurons,
          weights: updatedWeights,
        };
      });
    }, 400);

    return () => clearInterval(interval);
  }, []);

  if (!mlpData) return null;

  const layerColors = {
    input: 'blue',
    hidden: 'orange',
    output: 'purple',
  };

  return (
    <Canvas shadows>
      <OrbitControls />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} castShadow />
      {mlpData.neurons.map((neuron) => (
        <Node key={neuron.id} {...neuron} color={layerColors[neuron.layer]} />
      ))}
      {mlpData.weights.map((weight, index) => {
        const sourceNeuron = mlpData.neurons.find(n => n.id === weight.source);
        const targetNeuron = mlpData.neurons.find(n => n.id === weight.target);
        if (!sourceNeuron || !targetNeuron) {
          console.error(`Neuron not found for weight: ${weight}`);
          return null;
        }
        return (
          <Line
            key={index}
            points={[sourceNeuron.position, targetNeuron.position]}
            color={weight.color}
            lineWidth={weight.value * 4} // Adjust line width based on weight value
          />
        );
      })}
      <Text position={[-6, 3, 0]} fontSize={0.2} color="black">Input Layer</Text>
      <Text position={[-4, 3, 0]} fontSize={0.2} color="black">1st Hidden Layer</Text>
      <Text position={[0, 3, 0]} fontSize={0.2} color="black">2nd Hidden Layer</Text>
      <Text position={[3, 3, 0]} fontSize={0.2} color="black">Output Layer</Text>
    </Canvas>
  );
};

export default NeuralNetwork;