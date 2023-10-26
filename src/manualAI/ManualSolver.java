package manualAI;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mastermind.Combination;
import mastermind.Pin;
import mastermind.Response;

public class ManualSolver {
	
	private List<Combination> combinations = new ArrayList<Combination>();
	
	private int roundNumber = 0;
	
	private Combination lastGuess;
	private Combination secondLastGuess;
	
	public ManualSolver() {
		combinations = getEveryPossibleCombination();
	}
	
	public void reset() {
		roundNumber = 0;
		combinations = getEveryPossibleCombination();
	}
	
	public void processNewCombination(Combination combination, Response response, int roundNumber) {
		secondLastGuess = lastGuess;
		lastGuess = combination;
		int start_combinations = combinations.size();
		if(this.roundNumber + 1 != roundNumber) {
			System.out.println("Error while processing new combinations!");
			System.exit(-1);
		}
		this.roundNumber++;
		for(int i = combinations.size() - 1; i >= 0; i--) {
			if(!matches(combinations.get(i), combination, response)) {
				combinations.remove(i);
			}
		}
		System.out.println("Solver went through " + start_combinations + " combinations. Now there are " + combinations.size() + " possible combinations left.");
	}
	
	public List<Combination> getAvaiableCombinations(){
		return combinations;
	}
	
	public Combination guessCombination() {
		System.out.println("Computer is determining the best combination to guess...");
		if(combinations.size() <= 2) {
			return combinations.get(0);
		}
		if(roundNumber == 0) {
			return new Combination(Pin.A, Pin.B, Pin.C, Pin.D);
		}
		List<Combination> possibleGuesses = getEveryPossibleCombination();
		List<Response> possibleResponses = new ArrayList<Response>();
		possibleResponses.add(new Response(0, 1));
		possibleResponses.add(new Response(0, 2));
		possibleResponses.add(new Response(1, 0));
		possibleResponses.add(new Response(1, 1));
		possibleResponses.add(new Response(0, 0));
		possibleResponses.add(new Response(2, 0));
		possibleResponses.add(new Response(1, 2));
		possibleResponses.add(new Response(0, 3));
		possibleResponses.add(new Response(2, 1));
		possibleResponses.add(new Response(3, 0));
		
		List<Float> weights = Arrays.asList(new Float[] {0.9f, 0.63f, 0.597f, 0.549f, 0.511f, 0.213f, 0.142f, 0.119f, 0.0594f, 0.0264f});//avg. 6.088 with 2.2, min 3, max 10 in 500 rounds
		
		float bias = 2.2f;
		
		Combination bestCombination = new Combination(Pin.A, Pin.B, Pin.C, Pin.D);
		float greatestNumberOfEliminations = 0;
		for(Combination c: possibleGuesses) {
			float cNumberOfEliminations = 0;
			for(Combination d: combinations) {
				for(int i = 0; i < possibleResponses.size(); i++) {
					if(!matches(c, d, possibleResponses.get(i))) {
						cNumberOfEliminations += bias + weights.get(i);
					}
				}
			}
			if(cNumberOfEliminations > greatestNumberOfEliminations) {
				if(combinationMatchesCombination(c, lastGuess)) {
					continue;
				}
				if(secondLastGuess != null && combinationMatchesCombination(c, secondLastGuess)) {
					continue;
				}
				bestCombination = c;
				greatestNumberOfEliminations = cNumberOfEliminations;
			}
		}
		return bestCombination;
	}
	
	private boolean combinationMatchesCombination(Combination a, Combination b) {
		for(int i = 0; i < 4; i++) {
			if(a.getPins().get(i) != b.getPins().get(i)) {
				return false;
			}
		}
		return true;
	}
	
	private boolean matches(Combination a, Combination b, Response response) {
		Response r = a.compareToCombination(b);
		return r.isEquivalentToResponse(response);
	}
	
	private List<Combination> getEveryPossibleCombination() {
		int pinTypes = Pin.values().length;
		List<Combination> combinations = new ArrayList<Combination>();
		for(int i = 0; i < pinTypes; i++) {
			for(int j = 0; j < pinTypes; j++) {
				for(int k = 0; k < pinTypes; k++) {
					for(int l = 0; l < pinTypes; l++) {
						combinations.add(new Combination(Pin.values()[i], Pin.values()[j], Pin.values()[k], Pin.values()[l]));
					}
				}
			}
		}
		return combinations;
	}
	
	public void generateWeights() {
		List<Combination> a = getEveryPossibleCombination();
		List<Combination> b = getEveryPossibleCombination();
		Map<Response, Integer> possibleResponses = new HashMap<Response, Integer>();
		possibleResponses.put(new Response(0, 1), 0);
		possibleResponses.put(new Response(0, 2), 0);
		possibleResponses.put(new Response(1, 0), 0);
		possibleResponses.put(new Response(1, 1), 0);
		possibleResponses.put(new Response(0, 0), 0);
		possibleResponses.put(new Response(2, 0), 0);
		possibleResponses.put(new Response(1, 2), 0);
		possibleResponses.put(new Response(0, 3), 0);
		possibleResponses.put(new Response(2, 1), 0);
		possibleResponses.put(new Response(3, 0), 0);
		System.out.println("determining weights.");
		for(Combination c: a) {
			for(Combination d: b) {
				for(Response r: possibleResponses.keySet()) {
					if(matches(c, d, r)) {
						int responseNumber = possibleResponses.get(r);
						possibleResponses.replace(r, responseNumber + 1);
					}
				}
			}
		}
		for(Response response: possibleResponses.keySet()) {
			System.out.println(response.getResponseAsText() + ": " + possibleResponses.get(response));
		}
	}

}
