package inheritanceLearner;

public class ActivationFunctionTemplate {
	
	public float getActivationValue(float weightedInput) {
		System.out.println("[ERROR] default template activation function called");
		return weightedInput;
	}
	
	public float getActivationDerivative(float weightedInput) {
		System.out.println("[ERROR] default template derivative activation function called");
		return 0;
	}

}
