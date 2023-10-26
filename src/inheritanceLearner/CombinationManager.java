package inheritanceLearner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mastermind.Combination;
import mastermind.Pin;
import mastermind.Response;

public class CombinationManager {
	
	private static List<Combination> combinations = new ArrayList<Combination>();
	
	private static Random random = new Random();
	
	private static float[] DEFAULT_PROBABILITIES = new float[] {
		0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f,
		0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f,
		0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f,
		0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f
	};
	
	public static void reset() {
		combinations = getEveryPossibleCombination();
		System.out.println("CombinationManager, now: 4096.");
	}
	
	public static Combination getRandomCombination() {
		Pin a = Pin.values()[random.nextInt(8)];
		Pin b = Pin.values()[random.nextInt(8)];
		Pin c = Pin.values()[random.nextInt(8)];
		Pin d = Pin.values()[random.nextInt(8)];
		return new Combination(a, b, c, d);
	}
	
	public static boolean processNewCombination(Combination combination, Response response) {
		for(int i = combinations.size() - 1; i >= 0; i--) {
			if(!combinations.get(i).compareToCombination(combination).isEquivalentToResponse(response)) {
				combinations.remove(i);
			}
		}
		if(combinations.size() == 1) {
			return true;
		}
		System.out.println("CombinationManager, now: " + combinations.size() + ".");
		return false;
	}
	
	public static float[] getProbabilities(){
		if(combinations.size() == 4096) {
			return DEFAULT_PROBABILITIES;
		}
		float[] probabililties = new float[32];
		for(Combination c: combinations) {
			for(int i = 0; i < 4; i++) {
				Pin p = c.getPins().get(i);
				probabililties[i * 8 + p.ordinal()] += 1;
			}
		}
		for(int i = 0; i < 32; i++) {
			probabililties[i] = probabililties[i] / combinations.size();
		}
		return probabililties;
	}
	
	public static List<Combination> getAvaiableCombinations(){
		return combinations;
	}
	
	private static List<Combination> getEveryPossibleCombination() {
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

}
