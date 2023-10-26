package mastermind;

import java.util.Arrays;
import java.util.List;

public enum Pin {
	
	A, B, C, D, E, F, G, H;
	
	private static List<String> names = Arrays.asList(new String[] {"A", "B", "C", "D", "E", "F", "G", "H"});
	
	public static Pin getCorrespondingPin(String p) {
		int i = names.indexOf(p);
		if(i != -1) {
			return values()[i];
		}
		return null;
	}

}
