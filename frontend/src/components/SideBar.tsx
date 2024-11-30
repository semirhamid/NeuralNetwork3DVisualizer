import { ModelStructure } from './types';

// Sidebar Component
export const Sidebar: React.FC<{
  modelMetadata: ModelStructure;
  currentEpoch: number;
}> = ({ modelMetadata, currentEpoch }) => (
  <div className="w-80 bg-gray-800 text-white p-3 shadow-lg">
    <h2 className="text-xl font-semibold mb-4">Model Metadata</h2>
    <p className="mb-2">
      <strong className="text-gray-300">Total Layers:</strong>{' '}
      {modelMetadata.total_layers}
    </p>
    <p className="mb-2">
      <strong className="text-gray-300">Total Epoch:</strong>{' '}
      {modelMetadata.total_epochs}
    </p>
    <p className="mb-2">
      <strong className="text-gray-300">Current Epoch:</strong>
      <span className="text-red-700 font-extrabold">{currentEpoch}</span>
    </p>
    <p className="mb-2">
      <strong className="text-gray-300">Total Params:</strong>{' '}
      {modelMetadata.total_params}
    </p>
    <h3 className="text-lg font-medium mb-2">Layer Details:</h3>
    <ul className="space-y-3">
      {modelMetadata.layer_details.map((layer, index) => (
        <li
          key={`layer-meta-${index}`}
          className="bg-gray-700 p-3 rounded-lg shadow-md hover:bg-gray-600 transition flex space-x-2"
        >
          <strong className="block text-gray-100">{layer.layer_name}</strong>
          <span className="text-sm text-gray-400">
            Input: {layer.input_size}, Output: {layer.output_size}
          </span>
        </li>
      ))}
    </ul>
  </div>
);

// Slider Controls Component
export const SliderControls: React.FC<{
  layerSpacing: number;
  setLayerSpacing: (value: number) => void;
  neuronSpacing: number;
  setNeuronSpacing: (value: number) => void;
  nodeSize: number;
  setNodeSize: (value: number) => void;
}> = ({
  layerSpacing,
  setLayerSpacing,
  neuronSpacing,
  setNeuronSpacing,
  nodeSize,
  setNodeSize,
}) => (
  <div className="w-80 bg-gray-800 text-white h-screen p-4 shadow-lg">
    <h3 className="text-lg font-medium mb-4">Controls</h3>
    <div className="space-y-4">
      <div>
        <label className="block text-gray-300 mb-1">Layer Spacing</label>
        <input
          type="range"
          min="5"
          max="20"
          value={layerSpacing}
          onChange={(e) => setLayerSpacing(Number(e.target.value))}
          className="w-full cursor-pointer"
        />
      </div>
      <div>
        <label className="block text-gray-300 mb-1">Neuron Spacing</label>
        <input
          type="range"
          min="1"
          max="5"
          value={neuronSpacing}
          onChange={(e) => setNeuronSpacing(Number(e.target.value))}
          className="w-full cursor-pointer"
        />
      </div>
      <div>
        <label className="block text-gray-300 mb-1">Node Size</label>
        <input
          type="range"
          min="0.2"
          max="1"
          step="0.1"
          value={nodeSize}
          onChange={(e) => setNodeSize(Number(e.target.value))}
          className="w-full cursor-pointer"
        />
      </div>
    </div>
  </div>
);
