package inheritanceLearner;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
	
	List<Layer> layers = new ArrayList<Layer>();
	
	float[][][] weights;
	float[][] biases;
	
	int[] size;
	
	public NeuralNetwork(int[] size) {
		this.size = size;
		for(int i = 1; i < size.length; i++) {
			layers.add(new Layer(size[i - 1], size[i]));
		}
	}
	
	public NeuralNetwork(int[] size, float[][][] startingWeights, float[][] startingBiases) {
		for(int i = 1; i < size.length; i++) {
			layers.add(new Layer(size[i - 1], size[i], startingWeights[i - 1], startingBiases[i - 1]));
		}
		this.size = size;
		weights = startingWeights;
		biases = startingBiases;
	}

	public float[] evaluate(float[] inputs) {
		for(Layer layer: layers) {
			inputs = layer.evaluate(inputs);
		}
		return inputs;
	}
	
	public float[][][] getWeights(){
		if(weights != null) {
			return weights;
		}
		float[][][] r = new float[layers.size()][][];
		for(int i = 0; i < layers.size(); i++) {
			r[i] = layers.get(i).getWeights();
		}
		weights = r;
		return r;
	}
	
	public float[][] getBiases(){
		if(biases != null) {
			return biases;
		}
		float[][] r = new float[layers.size()][];
		for(int i = 0; i < layers.size(); i++) {
			r[i] = layers.get(i).getBiases();
		}
		biases = r;
		return r;
	}

	public int[] getSize() {
		return size;
	}

	public List<Layer> getLayers() {
		return layers;
	}
	
}
