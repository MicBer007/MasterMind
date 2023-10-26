package gradientDescentOnManualAI;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import log.Log;
import mastermind.Combination;
import mastermind.Pin;
import mastermind.Response;

public class GradientDescentAlgorithm {
	
	private static Log log = new Log("GDA", 6);
	
	private static final Combination[] POSSIBLE_COMBINATIONS = determinePossibleCombinations();
	
	private static final int MAX_COMBINATION_POOL_SIZE = 4096;
	
//	[GDA: 19962, INFO] Weight: 0.9
//	[GDA: 19962, INFO] Weight: 0.63
//	[GDA: 19962, INFO] Weight: 0.597
//	[GDA: 19962, INFO] Weight: 0.544361
//	[GDA: 19962, INFO] Weight: 0.511
//	[GDA: 19962, INFO] Weight: 0.210231
//	[GDA: 19962, INFO] Weight: 0.142
//	[GDA: 19962, INFO] Weight: 0.119
//	[GDA: 19962, INFO] Weight: 0.0594
//	[GDA: 19962, INFO] Weight: 0.0260568
//	[GDA: 19962, INFO] Weight: 0.0043
//	[GDA: 19962, INFO] Weight: 0.0048624002
//	[GDA: 19962, INFO] Weight: 0.004538025
//	[GDA: 19962, INFO] Weight: 0.0041
//	[GDA: 19962, INFO] Bias: 2.137914
//  percent change at the end: 0.00845
	
	private static final Response[] POSSIBLE_RESPONSES = new Response[] {
			new Response(0, 1),//1
			new Response(0, 2),//2
			new Response(1, 0),//4
			new Response(1, 1),//5
			new Response(0, 0),//0
			new Response(2, 0),//8
			new Response(1, 2),//6
			new Response(0, 3),//3
			new Response(2, 1),//9
			new Response(3, 0),//12
			new Response(0, 4),
			new Response(1, 3),
			new Response(2, 2),
			new Response(4, 0)
	};
	
	/*
	 * The following manually derived weight and bias values have solved with a 
	 * min of 3, max of 10, averaging 6.088 rounds per game, out of 500 games
	 */
	private static float[] weights = new float[] {
			0.9f, //W
			0.63f, //WW
			0.597f, //B
			0.549f, //BW
			0.511f, //
			0.213f, //BB
			0.142f, //BWW
			0.119f, //WWW
			0.0594f, //BBW
			0.0264f, //BBB
			0.0043f, //WWWW
			0.0048f, //BWWW
			0.0045f, //BBWW
			0.0041f //BBBB
	};
	
	private static float bias = 2.12f;
	
	private static float percentageChangeForWeights = 0.02f;
	
	private static float friction = 0.65f;
	
	public static void learn(int numberOfIterations) {
		float originalSuccess = evaluateCurrentWeights(createSamplePool(64));
		for(int i = 0; i < numberOfIterations; i++) {
			log.info("Starting new learning cycle! Original sucess is: " + originalSuccess, 10);
			float success = originalSuccess;
			//weights
			for(int weightID = 0; weightID < POSSIBLE_RESPONSES.length; weightID++) {
				float newSuccess = doGradientDescentOnWeight(weightID, true, originalSuccess);
				log.info("Finished checking up for weight number " + weightID + "! Success = " + newSuccess, 6);
				if(newSuccess < success) {
					success = newSuccess;
					continue; //Computer already found that going up was beneficial, and assumes that this will not be the case when going down (this assumption could be a bad thing)
				}
				success = doGradientDescentOnWeight(weightID, false, originalSuccess);
				log.info("Finished checking down for weight number " + weightID + "! Success = " + success, 6);
			}
			//bias value
			float newSucess = doGradientDescentOnBiasValue(true, originalSuccess);
			log.info("Finished checking up for bias value! Success = " + newSucess, 6);
			if(!(newSucess < originalSuccess)) { //Computer already found that going up was beneficial, and assumes that this will not be the case when going down (this assumption could be a bad thing)
				success = doGradientDescentOnBiasValue(false, originalSuccess);
				log.info("Finished checking down for bias value! Success = " + success, 6);
			}
			percentageChangeForWeights *= friction;
			log.info("Success now is: " + success + " Whereas original sucess was: " + originalSuccess);
			originalSuccess = success;
		}
		log.info("Learning cycles terminated.", 10);
		for(float weight: weights) {
			log.info("Weight: " + weight, 10);
		}
		log.info("Bias: " + bias, 10);
	}
	
	private static final int MAX_NUMBER_OF_ROUNDS_TO_TAKE = 16;
	
	public static float evaluateCurrentWeights(List<Combination> combinationPool) {
		log.info("Evaluating new combination pool!", 10);
		int totalRoundsTaken = 0;
		for(int i = 0; i < combinationPool.size(); i++) {
			totalRoundsTaken += determineNumberOfRoundsTakenGivenCurrentWeights(combinationPool.get(i));
		}
		return (float) (totalRoundsTaken) / (float) (combinationPool.size());
	}
	
	public static float evaluateCurrentWeightsToGivenLevelOfPrecision(List<Combination> combinationPool, float precision) {
		log.info("Evaluating new combination pool! Searching for precision of " + precision + ".", 10);
		float totalNumberOfRounds = 0;
		float numberGamesPlayed = 0;
		List<Float> averages = new ArrayList<Float>();
		for(; numberGamesPlayed < 8; numberGamesPlayed++) {
			float numberOfRounds = determineNumberOfRoundsTakenGivenCurrentWeights(combinationPool.get((int) numberGamesPlayed));
			totalNumberOfRounds += numberOfRounds;
			averages.add(totalNumberOfRounds / numberGamesPlayed);
		}
		for(; numberGamesPlayed < combinationPool.size(); numberGamesPlayed++) {
			float numberOfRounds = determineNumberOfRoundsTakenGivenCurrentWeights(combinationPool.get((int) numberGamesPlayed));
			totalNumberOfRounds += numberOfRounds;
			float average = totalNumberOfRounds / numberGamesPlayed;
			averages.add(average);
			if(hasEvaluationReachedCorrectDegreeOfPrecision(averages, precision)) { //if the results have reached the required precision
				log.info("Combination was evaluated for " + numberGamesPlayed + " games, and returned an average of " + averages.get(averages.size() - 1) + ".", 6);
				break;
			}
		}
		return totalNumberOfRounds / numberGamesPlayed;
	}
	
	private static boolean hasEvaluationReachedCorrectDegreeOfPrecision(List<Float> averages, float precision) {
		float currentAverage = averages.get(averages.size() - 1);
		for(int i = averages.size() - 2; i > averages.size() - 6; i--) {
			float difference = Math.abs(averages.get(i) - currentAverage);
			if(difference > precision) {
				log.info("Average is: " + currentAverage + ". Failed. difference = " + difference, 4);
				return false;
			}
		}
		log.info("Average is: " + currentAverage + ". Sucess!", 4);
		return true;
	}
	
	private static int determineNumberOfRoundsTakenGivenCurrentWeights(Combination solution) {
		log.info("Evaluating combination: " + solution.getCombinationAsText(), 2);
		List<Combination> combinationsLeft = new ArrayList<Combination> (Arrays.asList(POSSIBLE_COMBINATIONS.clone()));
		Combination lastGuess = null;
		Combination secondLastGuess = null;
		for(int roundNum = 1; roundNum < MAX_NUMBER_OF_ROUNDS_TO_TAKE; roundNum++) {
			Combination guess = makeGuess(roundNum, combinationsLeft, lastGuess, secondLastGuess);
			if(guess == solution) {
				log.info("Correct combination guessed in " + roundNum + " rounds.", 4);
				return roundNum;
			}
			secondLastGuess = lastGuess;
			lastGuess = guess;
			combinationsLeft = reduceCombinationsAccordingToGuess(combinationsLeft, guess, guess.compareToCombination(solution));
		}
		//if this part of the code is executed, no solution was found within MAX_NUMBER_OF_ROUNDS_TO_TAKE rounds.
		log.fatal("Exceeded max number of rounds!");
		return 16; //penalty TODO maybe consider making this penalty become greater as the presicion gets greater
	}
	
	private static List<Combination> reduceCombinationsAccordingToGuess(List<Combination> possibleCombinations, Combination guess, Response response) {
		for(int i = 0; i < possibleCombinations.size(); i++) {
			Combination combination = possibleCombinations.get(i);
			if(!combination.compareToCombination(guess).isEquivalentToResponse(response)) {
				possibleCombinations.remove(combination);
				i--;
			}
		}
		return possibleCombinations;
	}
	
	public static Combination makeGuess(int roundNumber, List<Combination> combinationsLeft, Combination lastGuess, Combination secondLastGuess) {
		log.info("Computer is determining the best combination to guess out of " + combinationsLeft.size() + " combinations.", 0);
		if(combinationsLeft.size() <= 2) {
			if(combinationsLeft.size() == 0) {
				log.fatal("There are no combinations left!");
				System.exit(-1);
			}
			log.info("Chose combination: " + combinationsLeft.get(0).getCombinationAsText(), 0);
			return combinationsLeft.get(0);
		}
		if(roundNumber == 1) {
			Combination defaultCombination = new Combination(Pin.A, Pin.B, Pin.C, Pin.D);
			log.info("Chose combination: " + defaultCombination.getCombinationAsText(), 0);
			return defaultCombination;
		}
		
		Combination bestCombination = null;
		float numberOfEliminations = 0;
		for(Combination possibleGuess: POSSIBLE_COMBINATIONS) {
			float cNumberOfEliminations = bias * 13 * combinationsLeft.size();
			for(Combination d: combinationsLeft) {
				Response r = d.compareToCombination(possibleGuess);
				for(int i = 0; i < POSSIBLE_RESPONSES.length; i++) {
					if(!POSSIBLE_RESPONSES[i].isEquivalentToResponse(r)) {
						cNumberOfEliminations += weights[i];
					}
				}
			}
			if(cNumberOfEliminations >= numberOfEliminations) {
				if(possibleGuess.isEquivalentToCombination(lastGuess)) {
					continue;
				}
				if(secondLastGuess != null && possibleGuess.isEquivalentToCombination(secondLastGuess)) {
					continue;
				}
				bestCombination = possibleGuess;
				numberOfEliminations = cNumberOfEliminations;
			}
		}
		if(bestCombination == null) {
			log.fatal("Function did not make a guess!");
			System.exit(-1);
		}
		log.info("Chose combination: " + bestCombination.getCombinationAsText(), 0);
		return bestCombination;
	}
	
	private static float doGradientDescentOnWeight(int weightID, boolean up, float previousEvaluation) {
		List<Combination> samplePool = createSamplePool(MAX_COMBINATION_POOL_SIZE);
		float originalWeight = weights[weightID];
		float change = originalWeight*percentageChangeForWeights;
		weights[weightID] += (up ? change : -change);
		float precision = percentageChangeForWeights;
		float newEvaluation = evaluateCurrentWeightsToGivenLevelOfPrecision(samplePool, precision);
		if(newEvaluation < previousEvaluation) {
			return newEvaluation; //and keep the weights in their new configuration
		} else {
			weights[weightID] -= (up ? change : -change); //reset the weight configuration
			return previousEvaluation;
		}
	}
	
	private static float doGradientDescentOnBiasValue(boolean up, float previousEvaluation) {
		List<Combination> samplePool = createSamplePool(MAX_COMBINATION_POOL_SIZE);
		float originalBias = bias;
		float change = originalBias*percentageChangeForWeights;
		bias += (up ? change : -change);
		float precision = percentageChangeForWeights;
		float newEvaluation = evaluateCurrentWeightsToGivenLevelOfPrecision(samplePool, precision);
		if(newEvaluation < previousEvaluation) {
			return newEvaluation;
		} else {
			bias -= (up ? change : -change);
			return previousEvaluation;
		}
	}
	
	private static List<Combination> createSamplePool(int samplePoolSize) {
		if(samplePoolSize <= 0 || samplePoolSize > MAX_COMBINATION_POOL_SIZE) return null;
		List<Combination> allCombinations = Arrays.asList(POSSIBLE_COMBINATIONS.clone());
		Collections.shuffle(allCombinations);
		List<Combination> samplePool = allCombinations.subList(0, samplePoolSize);
		if(samplePool.size() != samplePoolSize) {
			log.fatal("Sample pool is not the right size!");
			System.exit(-1);
		}
		return samplePool;
	}
	
	public static void saveInMemory(String fileName) { } //TODO
	
	public static Combination[] determinePossibleCombinations() {
		Combination[] combinations = new Combination[MAX_COMBINATION_POOL_SIZE];
		for(int i = 0; i < 8; i++) {
			for(int j = 0; j < 8; j++) {
				for(int k = 0; k < 8; k++) {
					for(int l = 0; l < 8; l++) {
						combinations[(((i * 8) + j) * 8 + k) * 8 + l] = new Combination(Pin.values()[i], Pin.values()[j], Pin.values()[k], Pin.values()[l]);
					}
				}
			}
		}
		if(!combinations[MAX_COMBINATION_POOL_SIZE - 1].compareToCombination(new Combination(Pin.H, Pin.H, Pin.H, Pin.H)).isSolved()) {
			System.out.println("method determinePossibleCombinations in GradientDescentAlgoritm failed!");
			System.exit(-1);
		}
		return combinations;
	}

}
