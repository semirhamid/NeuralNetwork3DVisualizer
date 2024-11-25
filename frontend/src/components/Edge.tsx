import React from 'react';

type EdgeProps = {
  source: [number, number, number];
  target: [number, number, number];
  value: number;
};

const Edge: React.FC<EdgeProps> = ({ source, target, value }) => {
  const color = value > 0 ? 'green' : 'red';
  const thickness = Math.abs(value) * 0.05;

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={new Float32Array([...source, ...target])}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} linewidth={thickness} />
    </line>
  );
};

export default Edge;
