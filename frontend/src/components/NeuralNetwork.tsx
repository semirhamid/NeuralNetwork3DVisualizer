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
      weights: [],
    };

    // Create dense connections between layers
    const inputNeurons = initialData.neurons.filter(n => n.layer === 'input');
    const hiddenNeurons = initialData.neurons.filter(n => n.layer === 'hidden');
    const outputNeurons = initialData.neurons.filter(n => n.layer === 'output');

    inputNeurons.forEach(inputNeuron => {
      hiddenNeurons.forEach(hiddenNeuron => {
        initialData.weights.push({
          source: inputNeuron.id,
          target: hiddenNeuron.id,
          value: Math.random(),
          color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
        });
      });
    });

    hiddenNeurons.forEach(hiddenNeuron => {
      outputNeurons.forEach(outputNeuron => {
        initialData.weights.push({
          source: hiddenNeuron.id,
          target: outputNeuron.id,
          value: Math.random(),
          color: `#${Math.floor(Math.random() * 16777215).toString(16)}`,
        });
      });
    });

    setMlpData(initialData);

    // Update the MLP data every 200 ms
    const interval = setInterval(() => {
      setMlpData(prevData => {
        if (!prevData) return null;

        // Update weights and neuron sizes
        const updatedNeurons = prevData.neurons.map(neuron => ({
          ...neuron,
          weight: Math.random() * 1.5 + 1, // Random weight between 1 and 4
        }));

        const updatedWeights = prevData.weights.map(weight => {
          const newValue = Math.random(); // New random weight value
          return {
            ...weight,
            value: weight.value + (newValue - weight.value) * 0.1, // Gradual change
            color: `#${Math.floor(Math.random() * 16777215).toString(16)}`, // Random color
          };
        });

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
    <Canvas shadows camera={{ fov: 50, position: [0, 0, 15] }}>
      <OrbitControls />
      <ambientLight intensity={0.7} />
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
            lineWidth={weight.value * 2} // Adjust line width based on weight value
          />
        );
      })}
      <Text position={[-6, 3, 0]} fontSize={0.2} fontWeight={"bold"} color="white">Input Layer</Text>
      <Text position={[-4, 3, 0]} fontSize={0.2} fontWeight={"bold"} color="white">1st Hidden Layer</Text>
      <Text position={[0, 3, 0]} fontSize={0.2} fontWeight={"bold"} color="white">2nd Hidden Layer</Text>
      <Text position={[3, 3, 0]} fontSize={0.2} fontWeight={"bold"} color="white">Output Layer</Text>
    </Canvas>
  );
};

export default NeuralNetwork;