import React from 'react';
import NeuralNetwork from './components/NeuralNetwork';
import './index.css';
import './App.css';
const App: React.FC = () => {
  return (
    <div className="app">
      <h1>Visualizing 2 Layer</h1>
      <div className="canvas-container">
        <NeuralNetwork />
      </div>
    </div>
  );
};

export default App;
