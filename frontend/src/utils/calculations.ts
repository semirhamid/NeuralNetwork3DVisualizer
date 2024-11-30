import { LayerData } from "../components/types";

export const calculateNeuronPositions = (
  mlpData: LayerData[],
  layerSpacing: number,
  neuronSpacing: number
): [number, number, number][][] => {
  return mlpData.map((layer, layerIndex) =>
    layer.weights.map((_, neuronIndex) => [
      layerIndex * layerSpacing,
      neuronIndex * neuronSpacing,
      0,
    ])
  );
};
