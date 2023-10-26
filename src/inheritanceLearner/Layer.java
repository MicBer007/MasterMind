package inheritanceLearner;

public class Layer {
		
	private Node[] nodes;
	
	private int numInputs;
	private int numOutputs;
	
	public Layer(int numInputs, int numOutputs) {
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
		nodes = new Node[numOutputs];
		for(int i = 0; i < numOutputs; i++) {
			nodes[i] = new Node(numInputs);
		}
	}
	
	public Layer(int numInputs, int numOutputs, float[][] startingWeights, float[] startingBiases) {
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
		nodes = new Node[numOutputs];
		for(int i = 0; i < numOutputs; i++) {
			nodes[i] = new Node(numInputs, startingWeights[i], startingBiases[i]);
		}
	}
	
	public float[] evaluate(float[] inputs){
		float[] outputs = new float[nodes.length];
		for(int i = 0; i < nodes.length; i++) {
			outputs[i] = nodes[i].evaluate(inputs);
		}
		return outputs;
	}
	
	public float[][] getWeights() {
		float[][] r = new float[numOutputs][numInputs];
		for(int i = 0; i < nodes.length; i++) {
			r[i] = nodes[i].getWeights();
		}
		return r;
	}
	
	public float[] getBiases() {
		float[] r = new float[numOutputs];
		for(int i = 0; i < nodes.length; i++) {
			r[i] = nodes[i].getBias();
		}
		return r;
	}
	
	public Node[] getNodes() {
		return nodes;
	}

	public int getNumInputs() {
		return numInputs;
	}

	public int getNumOutputs() {
		return numOutputs;
	}
	
}
