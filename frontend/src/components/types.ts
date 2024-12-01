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
  batch: number;
  batch_size: number;
  learning_rate: number;
  loss: number;
  layers: LayerData[];
  model_structure: ModelStructure;
}



export interface ModelMetaData {
  epoch: number;
  batch: number;
  batch_size: number;
  learning_rate: number;
  loss: number;
}