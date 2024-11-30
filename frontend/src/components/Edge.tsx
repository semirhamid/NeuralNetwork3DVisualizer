import React from 'react';
import { Line } from '@react-three/drei';

interface EdgeProps {
  sourcePosition: [number, number, number];
  targetPosition: [number, number, number];
  weight: number;
}

const Edge: React.FC<EdgeProps> = ({ sourcePosition, targetPosition, weight }) => {
  return (
    <Line
      points={[sourcePosition, targetPosition]}
      color={weight > 0 ? 'blue' : 'red'}
      lineWidth={Math.max(0.4, Math.abs(weight)) * 2}
    />
  );
};

export default Edge;
