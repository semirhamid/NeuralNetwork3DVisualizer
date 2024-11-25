import React, { useMemo } from 'react';
import { MeshProps } from '@react-three/fiber';
import * as THREE from 'three';

type NodeProps = MeshProps & {
  position: [number, number, number];
  weight: number;
  color: string;
};

const Node: React.FC<NodeProps> = ({ position, weight, color }) => {
  const texture = useMemo(() => {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');

    // Create gradient
    if (!context) return new THREE.CanvasTexture(canvas);
    
    const gradient = context.createRadialGradient(
      size / 2,
      size / 2,
      0,
      size / 2,
      size / 2,
      size / 2
    );
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.5, '#fefefe');
    gradient.addColorStop(1, color);

    // Fill with gradient
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);

    return new THREE.CanvasTexture(canvas);
  }, [color]);

  return (
    <mesh position={position} castShadow receiveShadow>
      <sphereGeometry args={[weight * 0.1, 48, 48]} />
      <meshStandardMaterial map={texture} />
    </mesh>
  );
};

export default Node;