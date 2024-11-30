import React from 'react';
import NeuralNetwork from './components/NeuralNetwork';
import './index.css';
const App: React.FC = () => {
  return (
    <div className="">
      <div className="canvas-container">
        <NeuralNetwork />
      </div>
    </div>
  );
};

export default App;
