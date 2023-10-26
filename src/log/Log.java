package log;

public class Log {
	
	private String className;
	
	private long startTimeMillis;
	
	private int logPriority;
	
	public Log(String className, int logLevel) {
		this.className = className;
		this.logPriority = logLevel;
		this.startTimeMillis = System.currentTimeMillis();
	}
	
	public Log(String className, LogLevel logLevel) {
		this.className = className;
		this.logPriority = logLevel.getLogValue();
		this.startTimeMillis = System.currentTimeMillis();
	}

	public String getClassName() {
		return className;
	}

	public int getLogLevel() {
		return logPriority;
	}

	public void setLogLevel(int logLevel) {
		this.logPriority = logLevel;
	}
	
	public long getSecsSinceStart() {
		return (long) (System.currentTimeMillis() - startTimeMillis) / 1000;
	}
	
	public void logRawMessage(String message, int logLevel) {
		if(logLevel >= this.logPriority) {
			System.out.println(message);
		}
	}
	
	public void log(String message, int logLevel) {
		logRawMessage("[" + className + ": " + getSecsSinceStart() + ", priority: " + logLevel + "] " + message , logLevel);
	}
	
	public void log(String message, LogLevel level) {
		logRawMessage("[" + className + ": " + getSecsSinceStart() + ", " + level.name() + "] " + message, level.getLogValue());
	}
	
	public void info(String message) {
		log(message, LogLevel.INFO);
	}
	
	public void warn(String message) {
		log(message, LogLevel.WARN);
	}
	
	public void error(String message) {
		log(message, LogLevel.ERROR);
	}
	
	public void fatal(String message) {
		log(message, LogLevel.FATAL);
	}
	
	public void log(String message, LogLevel level, int substitutePriority) {
		logRawMessage("[" + className + ": " + getSecsSinceStart() + ", " + level.name() + "] " + message, substitutePriority);
	}
	
	public void info(String message, int substitutePriority) {
		log(message, LogLevel.INFO, substitutePriority);
	}
	
	public void warn(String message, int substitutePriority) {
		log(message, LogLevel.WARN, substitutePriority);
	}
	
	public void error(String message, int substitutePriority) {
		log(message, LogLevel.ERROR, substitutePriority);
	}
	
	public void fatal(String message, int substitutePriority) {
		log(message, LogLevel.FATAL, substitutePriority);
	}

}
