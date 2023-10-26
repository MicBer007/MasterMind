package gradientDescentOnManualAI;

import java.util.Arrays;
import java.util.List;

import mastermind.Combination;
import mastermind.Pin;
import mastermind.Response;

public class Solver {
	
	private static int maxRounds = 20;
	
	private static final Combination[] ALL_COMBINATIONS = GradientDescentAlgorithm.determinePossibleCombinations();
	
	public static int solveCombination(float[] weights, float[] bias, Combination solution) {
		List<Combination> combinationsLeft = Arrays.asList(ALL_COMBINATIONS.clone());
		for(int roundNum = 0; roundNum < maxRounds; roundNum++) {
			if(combinationsLeft.size() <= 3) {
				Combination combinationGuessed = combinationsLeft.get(0);
				Response response = combinationGuessed.compareToCombination(solution);
				if(response.isSolved()) {
					return roundNum;
				}
				combinationsLeft = guess(combinationGuessed, response, combinationsLeft);
			}
			if(roundNum == 0) {
				Combination combinationGuessed = new Combination(Pin.A, Pin.B, Pin.C, Pin.D);
				Response response = combinationGuessed.compareToCombination(solution);
				if(response.isSolved()) {
					return -1;
				}
				combinationsLeft = guess(combinationGuessed, response, combinationsLeft);
				continue;
			}
			
		}
		System.out.println("Solver went into an infinite loop");
		System.exit(-1);
		return 1;
	}
	
	public static List<Combination> guess(Combination combinationGuessed, Response response, List<Combination> possibilities) {
		for(int i = 0; i < possibilities.size(); i++) {
			if(!combinationGuessed.compareToCombination(possibilities.get(i)).isEquivalentToResponse(response)) {
				possibilities.remove(i);
			}
		}
		return possibilities;
	}

}
