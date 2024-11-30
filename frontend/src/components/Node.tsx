import React from 'react';
import { Sphere } from '@react-three/drei';

interface NodeProps {
  position: [number, number, number];
  color: string;
  nodeSize: number;
}

const Node: React.FC<NodeProps> = ({ position, color, nodeSize }) => {
  return (
    <Sphere position={position} args={[nodeSize, 16, 16]}>
      <meshStandardMaterial color={color} />
    </Sphere>
  );
};

export default Node;
