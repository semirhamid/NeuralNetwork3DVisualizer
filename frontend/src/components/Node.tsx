import React from 'react';
import { Sphere } from '@react-three/drei';

interface NodeProps {
  position: [number, number, number];
  color: string;
  nodeSize: number;
  bias?: number | null; // Bias is optional
}

const Node: React.FC<NodeProps> = ({ position, color, nodeSize, bias }) => {
  const biasColor = bias
    ? bias < 0
      ? 'red' // Use red for negative biases
      : 'green' // Use green for positive biases
    : color;

  const biasSize = bias ? Math.abs(bias) * nodeSize : nodeSize;

  return bias !== null ? (
    <Sphere position={position} args={[biasSize, 16, 16]}>
      <meshStandardMaterial color={biasColor} />
    </Sphere>
  ) : (
    <Sphere position={position} args={[nodeSize, 16, 16]}>
      <meshStandardMaterial color={color} />
    </Sphere>
  );
};

export default Node;
