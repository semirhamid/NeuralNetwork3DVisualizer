// Types
export interface LayerData {
  layer: string;
  weights: number[][];
  biases: number[];
}

export interface ModelStructure {
  total_layers: number;
  total_epochs: number;
  total_params: number;
  layer_details: {
    layer_name: string;
    input_size: number;
    output_size: number;
  }[];
}

export interface TrainingData {
  epoch: number;
  layers: LayerData[];
  model_structure: ModelStructure;
}

