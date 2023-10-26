package redundantStuff;

import java.util.ArrayList;
import java.util.List;

import mastermind.Combination;
import mastermind.Pin;

public enum PossibleBaseCombination {
	
	ABCD(new Combination(Pin.A, Pin.B, Pin.C, Pin.D)),
	AABC(new Combination(Pin.A, Pin.A, Pin.B, Pin.C)),
	AABB(new Combination(Pin.A, Pin.A, Pin.B, Pin.B)),
	AAAB(new Combination(Pin.A, Pin.A, Pin.A, Pin.B)),
	AAAA(new Combination(Pin.A, Pin.A, Pin.A, Pin.A));
	
	private Combination combination;
	
	private PossibleBaseCombination(Combination combination) {
		this.combination = combination;
	}
	
	public Combination getCombination() {
		return combination;
	}
	
	public static PossibleBaseReturn evaluateCombination(Combination combination) {
		PossibleBaseReturn r = new PossibleBaseReturn();
		int highest = 0;
		int[] locations = new int[4];
		for(int i = 0; i < Pin.values().length; i++) {
			Pin pin = Pin.values()[i];
			int numOfThisType = 0;
			List<Integer> typeLocations = new ArrayList<Integer>();
			for(int j = 0; j < combination.getPins().size(); j++) {
				if(pin == combination.getPins().get(j)) {
					numOfThisType++;
					typeLocations.add(j);
				}
			}
			if(numOfThisType > highest) {
				highest = numOfThisType;
				for(int l = 0; l < typeLocations.size(); l++) {
					locations[l] = typeLocations.get(l);
				}
			}
		}
		if(highest == 1) {
			r.ordering = new int[] {0, 1, 2, 3};
			r.baseCombination = PossibleBaseCombination.ABCD;
			return r;
		} else if(highest == 3) {
			int other = 0;
			//the following function assumes that the integers in locations are sorted from least to greatest
			for(int i = 0; i < highest; i++) {
				if(locations[i] == other) {
					other++;
				}
			}
			r.ordering = new int[] {locations[0], locations[1], locations[2], other};
			r.baseCombination = PossibleBaseCombination.AAAB;
			return r;
		} else if(highest == 4) {
			r.ordering = new int[] {0, 1, 2, 3};
			r.baseCombination = PossibleBaseCombination.AAAA;
			return r;
		}
		
		int secondHighest = 0;
		int[] secondLocations = new int[4];
		for(int i = 0; i < Pin.values().length; i++) {
			if(i == combination.getPins().get(locations[0]).ordinal()) {
				continue;
			}
			Pin pin = Pin.values()[i];
			int numOfThisType = 0;
			List<Integer> typeLocations = new ArrayList<Integer>();
			for(int j = 0; j < combination.getPins().size(); j++) {
				if(pin == combination.getPins().get(j)) {
					numOfThisType++;
					typeLocations.add(j);
				}
			}
			if(numOfThisType > secondHighest) {
				secondHighest = numOfThisType;
				for(int l = 0; l < typeLocations.size(); l++) {
					secondLocations[l] = typeLocations.get(l);
				}
			}
		}
		if(secondHighest == 2) {
			r.ordering = new int[] {locations[0], locations[1], secondLocations[0], secondLocations[1]};
			r.baseCombination = PossibleBaseCombination.AABB;
			return r;
		} else {
			int other = 0;
			//the following function assumes that the integers in locations are sorted from least to greatest
			for(int i = 0; i < highest; i++) {
				if(locations[i] == other) {
					other++;
				}
				if(secondLocations[0] == other) {
					i--;
					other++;
				}
			}
			r.ordering = new int[] {locations[0], locations[1], secondLocations[0], other};
			r.baseCombination = PossibleBaseCombination.AABC;
			return r;
		}
	}

}
