package mastermind;

import java.util.ArrayList;
import java.util.List;

public class Response {
	
	private int numBlackPins;
	private int numWhitePins;
	
	public Response(int numBlackPins, int numWhitePins) {
		this.numBlackPins = numBlackPins;
		this.numWhitePins = numWhitePins;
	}
	
	public int getNumBlackPins() {
		return numBlackPins;
	}
	
	public int getNumWhitePins() {
		return numWhitePins;
	}
	
	public String getResponseAsText() {
		String text = "";
		for(int i = 0; i < numBlackPins; i++) {
			text += "X";
		}
		for(int i = 0; i < numWhitePins; i++) {
			text += "O";
		}
		return text;
	}
	
	public boolean isEquivalentToResponse(Response otherResponse) {
		return isResponseEquivalentToResponse(this, otherResponse);
	}
	
	public static boolean isResponseEquivalentToResponse(Response a, Response b) {
		return a.getNumBlackPins() == b.getNumBlackPins() && a.getNumWhitePins() == b.getNumWhitePins();
	}
	
	public boolean isSolved() {
		return numBlackPins == 4;
	}
	
	public static boolean isComparisonBetweenCombinationsEquivalentToResponse(Combination a, Combination b, Response r) {
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
		return r.getNumBlackPins() == numBlackPins && r.getNumWhitePins() == numWhitePins;
//		List<Pin> aPins = new ArrayList<Pin>();
//		aPins.addAll(a.getPins());
//		List<Pin> bPins = new ArrayList<Pin>();
//		bPins.addAll(b.getPins());
//		int numBlackPinsFound = 0;
//		for(int i = 0; i < 4; i++) {
//			if(a.getPins().get(i - numBlackPinsFound) == b.getPins().get(i - numBlackPinsFound)) {
//				aPins.remove(i - numBlackPinsFound);
//				bPins.remove(i - numBlackPinsFound);
//				numBlackPinsFound++;
//			}
//		}
//		if(numBlackPinsFound != r.getNumBlackPins()) {
//			return false;
//		}
//		int numWhitePinsFound = 0;
//		for(Pin pin: bPins) {
//			for(int j = 0; j < aPins.size(); j++) {
//				if(pin == aPins.get(j)) {
//					numWhitePinsFound++;
//					aPins.remove(j);
//					break;
//				}
//			}
//		}
//		return numWhitePinsFound == r.getNumWhitePins();
	}

}
