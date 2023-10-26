package inheritanceLearner;

import main.Settings;

public class Node {
	
	private int numInputs;
	
	private float[] weights;
	
	private float bias;
	
	public Node(int numInputs) {
		this.numInputs = numInputs;
		weights = new float[numInputs];
		for(int i = 0; i < numInputs; i++) {
			weights[i] = Settings.DEFAULT_WEIGTH_VALUE;
		}
		bias = Settings.DEFAULT_BIAS_VALUE;
	}
	
	public Node(int numInputs, float[] startingWeights, float startingBias) {
		this.numInputs = numInputs;
		weights = startingWeights;
		bias = startingBias;
	}
	
	public float evaluate(float[] inputs) {
		float output = bias;
		for(int i = 0; i < inputs.length; i++) {
			output += inputs[i];
		}
		return Settings.ACTIVATION_FUNCTION.getActivationValue(output);
	}
	
	
	//getters
	public int getNumInputs() {
		return numInputs;
	}

	public float[] getWeights() {
		return weights;
	}

	public float getBias() {
		return bias;
	}

	public void setBias(float bias) {
		this.bias = bias;
	}
	
}