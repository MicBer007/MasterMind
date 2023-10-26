package inheritanceLearner;

public enum ActivationFunction {
	
	STEP(new ActivationFunctionTemplate() {
		@Override
		public float getActivationValue(float weightedInput) {
			return (weightedInput > 0) ? 1 : 0;
		}
	}),
	SIGMOID(new ActivationFunctionTemplate() {
		@Override
		public float getActivationValue(float weightedInput) {
			return (float) (1 / (1 + (Math.pow(Math.E, -weightedInput))));
		}
	});
	
	private ActivationFunctionTemplate i;
	
	private ActivationFunction(ActivationFunctionTemplate i) {
		this.i = i;
	}
	
	public float getActivationValue(float weightedInput) {
		return i.getActivationValue(weightedInput);
	}
	
	public float getActivationDerivative(float weightedInput) {
		return i.getActivationDerivative(weightedInput);
	}

}
