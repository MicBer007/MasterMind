package redundantStuff;

import mastermind.Combination;
import mastermind.Pin;
import mastermind.Response;

public class PinProbabilities {
	
	public static final Response[] availableResponses = new Response[] {
			new Response(0, 0),
			new Response(0, 1),
			new Response(1, 0),
			new Response(0, 2),
			new Response(1, 1),
			new Response(2, 0),
			new Response(0, 3),
			new Response(1, 2),
			new Response(2, 1),
			new Response(3, 0),
			new Response(0, 4),
			new Response(1, 3),
			new Response(2, 2)
	};
	
	public static float[][] determineProbabilitiesForBaseCombination(Combination baseComb, Response response) {
		float[][] percentages = new float[4][4];
		
		int otherPossibilities = 0;
		
		for(int i = 0; i < 8; i++) {
			for(int j = 0; j < 8; j++) {
				for(int k = 0; k < 8; k++) {
					for(int l = 0; l < 8; l++) {
						Combination ijkl = new Combination(Pin.values()[i], Pin.values()[j], Pin.values()[k], Pin.values()[l]);
						if(baseComb.compareToCombination(ijkl).isEquivalentToResponse(response)) {
							otherPossibilities++;
							for(int x = 0; x < 4; x++) {
								for(int y = 0; y < 4; y++) {
									if(baseComb.getPins().get(x) == ijkl.getPins().get(y)) {
										percentages[x][y] += 1;
									}
								}
							}
						}
					}
				}
			}
		}
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				percentages[i][j] = percentages[i][j] / otherPossibilities;
			}
		}
		return percentages;
	}

}
