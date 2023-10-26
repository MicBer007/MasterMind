package log;

import java.util.Arrays;
import java.util.List;

public enum LogLevel {
	
	INFO, WARN, ERROR, FATAL;
	
	private List<Integer> logValues = Arrays.asList(new Integer[] {4, 6, 8, 10});
	
	public int getLogValue() {
		return logValues.get(ordinal());
	}

}
