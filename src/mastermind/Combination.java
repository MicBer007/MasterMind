package mastermind;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Combination {
	
	private List<Pin> pins = new ArrayList<Pin>();
	
	public Combination(List<Pin> pins) {
		if(pins.size() == 4) {
			this.pins = pins;
		}
	}
	
	public Combination(Pin a, Pin b, Pin c, Pin d) {
		pins = Arrays.asList(new Pin[] {a, b, c, d});
	}
	
	public List<Pin> getPins(){
		return pins;
	}
	
	public String getCombinationAsText() {
		return pins.get(0).name() + pins.get(1).name() + pins.get(2).name() + pins.get(3).name();
	}
	
	public static Response compareCombinationToCombination(Combination a, Combination b) {
		List<Pin> aPins = new ArrayList<Pin>();
		aPins.addAll(a.getPins());
		List<Pin> bPins = new ArrayList<Pin>();
		bPins.addAll(b.getPins());
		int numBlackPins = 0;
		for(int i = 0; i < 4; i++) {
			if(aPins.get(i - numBlackPins) == bPins.get(i - numBlackPins)) {
				aPins.remove(i - numBlackPins);
				bPins.remove(i - numBlackPins);
				numBlackPins++;
			}
		}
		int numWhitePins = 0;
		for(Pin pin: bPins) {
			for(int j = 0; j < aPins.size(); j++) {
				if(pin == aPins.get(j)) {
					numWhitePins++;
					aPins.remove(j);
					break;
				}
			}
		}
		return new Response(numBlackPins, numWhitePins);
	}
	
	public Response compareToCombination(Combination combination) {
		return compareCombinationToCombination(this, combination);
	}
	
	public static boolean isCombinationEquivalentToCombination(Combination a, Combination b) {
		for(int i = 0; i < 4; i++) {
			if(a.getPins().get(i) != b.getPins().get(i)) {
				return false;
			}
		}
		return true;
	}
	
	public boolean isEquivalentToCombination(Combination otherCombination) {
		return isCombinationEquivalentToCombination(this, otherCombination);
	}

}
