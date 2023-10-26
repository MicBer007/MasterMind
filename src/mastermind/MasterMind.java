package mastermind;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import manualAI.ManualSolver;

public class MasterMind {
	
	public List<Combination> combinations = new ArrayList<Combination>();
	
	public List<Response> responses = new ArrayList<Response>();
	
	public Combination answer;
	
	public ManualSolver solver;
	
	public Random random;
	
	public Scanner scanner;
	
	public boolean solved;
	
	public MasterMind() {
		random = new Random();
		scanner = new Scanner(System.in);
		solver = new ManualSolver();
		System.out.println("All Mastermind pin combinations must come in the form of four uppercase characters from A to H. eg. 'ABGE'");
		System.out.print("Would you like to solve a Mastermind, or would you like the computer to do it for you? (M/C): ");
		String response = scanner.nextLine();
		if(response.equalsIgnoreCase("M")) {
			runAsPlayer();
		} else if(response.equalsIgnoreCase("C")){
			runAsComputer();
		}
	}
	
	public void runAsComputer() {
		initSolvingInstance();
		solver.reset();
		System.out.print("Please type a combination that the computer will try to solve:");
		answer = requestCombination();
		while(!solved) {
			processCombination(solver.guessCombination(), true);
		}
		System.out.println("The computer solved the combination in " + combinations.size() + " guesses.");
	}
	
	public void runAsPlayer() {
		initSolvingInstance();
		answer = randomizeCombination();
		requestPermissionToDisplayAnswer();
		while(!solved) {
			System.out.println("Please guess a combination.");
			processCombination(requestCombination(), false);
		}
		System.out.println("Congratulations! You solved the puzzle in " + combinations.size() + " turn" + (combinations.size() == 1 ? "!":"s!"));
	}
	
	private void initSolvingInstance() {
		combinations.clear();
		responses.clear();
		solved = false;
	}
	
	private void processCombination(Combination combination, boolean showComputer) {
		Response response = answer.compareToCombination(combination);
		combinations.add(combination);
		responses.add(response);
		
		if(showComputer) {
			solver.processNewCombination(combination, response, combinations.size());
		}
			
		printSummary();
		
		if(response.isSolved()) {
			solved = true;
		}
	}
	
	public void requestPermissionToDisplayAnswer() {
		System.out.print("Would you like to see the answer beforehand? [y/N]: ");
		String response = scanner.nextLine();
		if(response.equalsIgnoreCase("y")) {
			System.out.print("The answer is: " + answer.getCombinationAsText());
		}
	}
	
	public Combination randomizeCombination() {
		Pin a = Pin.values()[random.nextInt(8)];
		Pin b = Pin.values()[random.nextInt(8)];
		Pin c = Pin.values()[random.nextInt(8)];
		Pin d = Pin.values()[random.nextInt(8)];
		return new Combination(a, b, c, d);
	}
	
	public Combination requestCombination() {
		String input = scanner.nextLine();
		if(input.length() < 4) {
			System.out.println("This is not a valid combination! Combinations must have 4 uppercase characters from A to H!");
			return requestCombination();
		}
		Pin a = Pin.getCorrespondingPin(input.substring(0, 1));
		Pin b = Pin.getCorrespondingPin(input.substring(1, 2));
		Pin c = Pin.getCorrespondingPin(input.substring(2, 3));
		Pin d = Pin.getCorrespondingPin(input.substring(3, 4));
		if(a == null || b == null || c == null || d == null) {
			System.out.println("This is not a valid combination! Combinations must have 4 uppercase characters from A to H!");
			return requestCombination();
		}
		return new Combination(a, b, c, d);
	}
	
	public void printSummary() {
		System.out.println("\nHere are the results:");
		for(int i = 0; i < combinations.size(); i++) {
			if(combinations.size() > i) {
				System.out.print(combinations.get(i).getCombinationAsText() + ": " + responses.get(i).getResponseAsText());
			}
		}
		System.out.println();
	}

}
