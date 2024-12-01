import React from 'react';
import NeuralNetwork from './components/NeuralNetwork';
import './index.css';
import ErrorBoundary from './components/ErrorBoundary';
const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <div className="canvas-container">
        <NeuralNetwork />
      </div>
    </ErrorBoundary>
  );
};

export default App;
