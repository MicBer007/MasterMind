package main;

import gradientDescentOnManualAI.GradientDescentAlgorithm;
import mastermind.MasterMind;

public class Main {

	public static void main(String[] args) {
		GradientDescentAlgorithm.learn(3);
	}
	
	public static void doMasterMind() {
		new MasterMind();
	}

}
