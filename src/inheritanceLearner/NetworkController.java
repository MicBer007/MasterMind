package inheritanceLearner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mastermind.Combination;
import mastermind.Pin;
import mastermind.Response;

public class NetworkController {
	
	public static final int START_NUMBER_OF_NETWORKS = 7;
	
	public static final int NUMBER_OF_OFFSPRING_PER_PARENT = 7;
	
	public static final int NUMBER_OF_SUCCESSFUL_GENERATIONS_TO_PURSUE = 2;
	
	public static final int[] NETWORK_SIZE = new int[] {32, 70, 32};
	
	public static final int NUMBER_OF_MINDMASTER_GAMES_TO_RUN_PER_NETWORK = 3;
	
	public static final int NUMBER_OF_ROUNDS_TO_PLAY_UNTIL_TERMINATION = 5;
	
	public static float startDistributionRangeForWeights = 2f;
//	public static float minStartDistributionRangeForWeights = 1;
	public static float startDistributionRangeForBiases = 2f;
//	public static float minStartDistributionRangeForBiases = 1;
	
	public static final float STARTING_VELOCITY = 1;
	
	public static float velocity = STARTING_VELOCITY;
	public static float friction = 1f;
	
//	public static final float END_VELOCITY = 0.01f;
	public static final int NUMBER_OF_LEARNING_ITERATIONS_TO_GO_THROUGH = 2;
	
	public static Random r = new Random();
	
	public static List<NeuralNetwork> networks = new ArrayList<NeuralNetwork>();
	
	public static void startLearningProcedure(boolean startRecurringIteration) {
		System.out.println("Starting neural process!");
		NeuralNetwork network = new NeuralNetwork(NETWORK_SIZE);
		for(int i = 0; i < START_NUMBER_OF_NETWORKS; i++) {
			networks.add(generateNeuralNetworkBasedOnProgenitor(network));
		}
		if(startRecurringIteration) {
			iterateLearningProcedure(true, NUMBER_OF_LEARNING_ITERATIONS_TO_GO_THROUGH);
		}
	}
	
	public static void iterateLearningProcedure(boolean recur, int numberIterationsLeft) {
		System.out.println("Starting new generation!");
		if(numberIterationsLeft == 0) {
			endLearningProcedure();
			return;
		}
//		if(velocity < END_VELOCITY) {
//			endLearningProcedure();
//			return;
//		}
		List<NeuralNetwork> successfulNetworks = getSucessfulNetworks(NUMBER_OF_SUCCESSFUL_GENERATIONS_TO_PURSUE);
		networks.clear();
		for(int i = 0; i < NUMBER_OF_SUCCESSFUL_GENERATIONS_TO_PURSUE; i++) {
			for(int j = 0; j < NUMBER_OF_OFFSPRING_PER_PARENT; j++) {
				networks.add(generateNeuralNetworkBasedOnProgenitor(successfulNetworks.get(i)));
			}
		}
//		networks.add(successfulNetworks.get(0));
//		networks.add(successfulNetworks.get(1));
		velocity = velocity * friction;
		if(recur) {
			iterateLearningProcedure(true, numberIterationsLeft - 1);
		}
	}
	
	public static void endLearningProcedure() {
		NeuralNetwork bestNetwork = getSucessfulNetworks(1).get(0);
//		printNeuralNetwork(bestNetwork);
		System.out.println("Network learning completed!");
		CombinationManager.reset();
		System.out.println("----------------");
		Combination solution = CombinationManager.getRandomCombination();
		System.out.println(solution.getCombinationAsText());
		for(int j = 0; j < NUMBER_OF_ROUNDS_TO_PLAY_UNTIL_TERMINATION; j++) {
			float[] networkOutput = bestNetwork.evaluate(CombinationManager.getProbabilities());
			for(float f: networkOutput) {
				System.out.print(f + ", ");
			}
			System.out.println();
			Combination guess = interpretNeuralNetworkGuess(networkOutput);
			System.out.println(guess.getCombinationAsText());
			Response response = solution.compareToCombination(guess);
			if(CombinationManager.processNewCombination(guess, response)) { //if won
				break;
			}
		}
		
		//store final weights and biases
	}
	
	private static void printNeuralNetwork(NeuralNetwork network) {
		System.out.println("Printing out neural network.");
		for(int i = 0; i < network.getLayers().size(); i++) {
			Layer layer = network.getLayers().get(i);
			System.out.println("---------  LAYER " + (i + 1) + "  ----------");
			for(int j = 0; j < layer.getNodes().length; j++) {
				Node node = layer.getNodes()[j];
				String line = "#" + j + " weights: (";
				for(int k = 0; k < node.getWeights().length; k++) {
					line += node.getWeights()[k] + ", ";
				}
				line = line.substring(0, line.length() - 2) + ")";
				System.out.println(line);
				System.out.println("#" + j + " Bias: " + node.getBias());
			}
		}
	}
	
	private static List<NeuralNetwork> getSucessfulNetworks(int amountSuccessfulNeeded) {
		List<Float> roundsUsed = new ArrayList<Float>();
		List<Float> combinationsLeft = new ArrayList<Float>();
		for(int n = 0; n < networks.size(); n++) {
			NeuralNetwork network = networks.get(n);
			int totalRoundsUsed = 0;
			int totalCombinationsLeft = 0;
			for(int i = 0; i < NUMBER_OF_MINDMASTER_GAMES_TO_RUN_PER_NETWORK; i++) {
				CombinationManager.reset();
				Combination solution = new Combination(Pin.A, Pin.A, Pin.A, Pin.A);//CombinationManager.getRandomCombination();
				boolean terminated = true;
				for(int j = 0; j < NUMBER_OF_ROUNDS_TO_PLAY_UNTIL_TERMINATION; j++) {
					Combination guess = interpretNeuralNetworkGuess(network.evaluate(CombinationManager.getProbabilities()));
					Response response = solution.compareToCombination(guess);
					if(CombinationManager.processNewCombination(guess, response)) { //if won
						totalRoundsUsed += j;
						terminated = false;
						break;
					}
				}
				if(terminated) {
					totalRoundsUsed += NUMBER_OF_ROUNDS_TO_PLAY_UNTIL_TERMINATION;
					totalCombinationsLeft += CombinationManager.getAvaiableCombinations().size();
				}
			}
			float averageR = totalRoundsUsed / (float) NUMBER_OF_MINDMASTER_GAMES_TO_RUN_PER_NETWORK;
			float averageCL = totalCombinationsLeft / (float) NUMBER_OF_MINDMASTER_GAMES_TO_RUN_PER_NETWORK;
			roundsUsed.add(averageR);
			combinationsLeft.add(averageCL);

			System.out.println("finished processing network #" + (n + 1) + " out of " + networks.size() + " TR = " + averageR + ", CL = " + averageCL);
		}
		
		roundsUsed.sort((Float a, Float b) -> a.compareTo(b));
		combinationsLeft.sort((Float a, Float b) -> a.compareTo(b));
		List<Integer> indicesOfSuccessfulGenerations = new ArrayList<Integer>();
		for(int i = 0; i < NUMBER_OF_SUCCESSFUL_GENERATIONS_TO_PURSUE; i++) {
			float highestNumberRoundsUsed = 0;
			int index = -1;
			for(int j = 0; j < networks.size(); j++) {
				if(indicesOfSuccessfulGenerations.contains(j)) continue;
				if(roundsUsed.get(j) > highestNumberRoundsUsed) {
					highestNumberRoundsUsed = roundsUsed.get(j);
					index = j;
				}
			}
			
			if(!(highestNumberRoundsUsed >= NUMBER_OF_ROUNDS_TO_PLAY_UNTIL_TERMINATION)) {
				if(index == -1) {
					continue; //throw new IllegalArgumentException();
				}
				indicesOfSuccessfulGenerations.add(index);
				continue;
			}

			float highestNumberCombinationsLeft = 0;
			for(int j = 0; j < networks.size(); j++) {
				if(indicesOfSuccessfulGenerations.contains(j)) continue;
				if(combinationsLeft.get(j) > highestNumberCombinationsLeft) {
					highestNumberCombinationsLeft = combinationsLeft.get(j);
					index = j;
				}
			}
			if(index == -1) {
				throw new IllegalArgumentException();
			}
			indicesOfSuccessfulGenerations.add(index);
		}
		List<NeuralNetwork> successfulNetworks = new ArrayList<NeuralNetwork>();
		for(Integer i: indicesOfSuccessfulGenerations) {
			successfulNetworks.add(networks.get(i));
		}
		System.out.println(successfulNetworks.size());
		if(successfulNetworks.size() != NUMBER_OF_SUCCESSFUL_GENERATIONS_TO_PURSUE) throw new IllegalArgumentException();
		return successfulNetworks;
	}
	
	private static NeuralNetwork generateNeuralNetworkBasedOnProgenitor(NeuralNetwork progenitor) {
		float velocityForWeights = velocity * startDistributionRangeForWeights;
		
		//randomizing new weights
		float[][][] startingWeights = progenitor.getWeights();
		for(int i = 0; i < startingWeights.length; i++) {
			for(int j = 0; j < startingWeights[i].length; j++) {
				for(int k = 0; k < startingWeights[i][j].length; k++) {
					startingWeights[i][j][k] = startingWeights[i][j][k] + ((r.nextFloat() - 0.5f) * 2 * velocityForWeights);
				}
			}
		}

		float velocityForBiases = velocity * startDistributionRangeForBiases;
		
		//randomizing new biases
		float[][] startingBiases = progenitor.getBiases();
		for(int i = 0; i < startingBiases.length; i++) {
			for(int j = 0; j < startingBiases[i].length; j++) {
				startingBiases[i][j] = startingBiases[i][j] + ((r.nextFloat() - 0.5f) * 2 * velocityForBiases);
			}
		}
		
		return new NeuralNetwork(progenitor.size, startingWeights, startingBiases);
	}
	
	private static Combination interpretNeuralNetworkGuess(float[] outputs) {
		Pin[] pins = new Pin[4];
		for(int i = 0; i < 4; i++) {
			int highestOrdinal = -1;
			float preference = -1;
			for(int j = 0; j < 8; j++) {
				if(outputs[i * 8 + j] > preference) {
					preference = outputs[i * 8 + j];
					highestOrdinal = j;
				}
			}
			pins[i] = Pin.values()[highestOrdinal];
		}
		return new Combination(pins[0], pins[1], pins[2], pins[3]);
	}

}
