package artificial_neural_networks;

import java.io.*;
import java.util.*;


public class Tester {
	double[] InputNode;
	double[] HiddenNode;
	double[] OutputNode;
	double LearningRate;
	final String TrainFileName = "mnist_train.csv";
	final String TestFileName = "mnist_test.csv";
	List<String> TrainCases;
	List<String> TestCases;
	double[][] InputHidden;
	double[][] HiddenOutput;
	double[] OutputError;
	double[] HiddenError;
	double[] TargetResults = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,};
	int test_no;
	int test_pass;

	public void initialize (int inputnodes, int hiddennodes, int outputnodes, double Learningrate)
	{
		InputNode = new double[inputnodes + 1];//inputnodes;			//784
		HiddenNode = new double[hiddennodes];			//100
		OutputNode = new double[outputnodes];			// 10
		LearningRate = Learningrate;			// 0.3
		TrainCases = new ArrayList<>();
		TestCases = new ArrayList<>();
		InputHidden = new double [inputnodes][hiddennodes];
		HiddenOutput = new double [hiddennodes][outputnodes];
		HiddenError = new double [hiddennodes];
		OutputError = new double [outputnodes];

		test_pass = 0;
		test_no = 0;

		RandomlizeWeights(InputHidden);
		RandomlizeWeights(HiddenOutput);
	}

	public void train(int nCases)
	{
		ReadTrainCases(nCases);
		for(int i = 0; i < TrainCases.size(); i++) {
			TrainCase(TrainCases.get(i));
			System.out.println("Case " + i + " done!");
		}
	}

	public void query()
	{
		QueryStepOne();
		QueryStepTwo();

	}

	public void query(int nCases)
	{
		ReadTestCases(nCases);
		for(int i = 0; i < TestCases.size(); i++) {
			RunTest(TestCases.get(i));
		}
		test_no = nCases;
		System.out.println("total tests : " + test_no);
		System.out.println("total pass : " + test_pass);

	}

	private void RunTest(String string) {
		List<Double> result = StringToArray(string);
		double target = result.get(0);
		TargetResults = new double[] {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,}; //reset
		TargetResults[(int) target] = 0.99;
		RescaleWeights(result);

		for (int i = 0; i < InputNode.length; i++){
			InputNode[i] = result.get(i);
		}

		query();

		double max = 0;
		int id = -1;
		for (int i = 0; i < OutputNode.length; i++) {
			if (OutputNode[i] >= max) {
				max = OutputNode[i];
				id = i;
			}
		}
		System.out.println("Output: " + id + "   Actual: " + target);
		if (id == target) {
			test_pass++;
		}

	}

	private void QueryStepOne()
	{
		for (int i = 0; i < HiddenNode.length; i++){
			for(int j = 0; j < InputHidden.length; j++) {
				HiddenNode [i] = HiddenNode [i] + InputNode[j + 1] * InputHidden[j][i];
			}
		}

		for (int i = 0; i < HiddenNode.length; i++) {
			HiddenNode[i] = sigmoid(HiddenNode[i]);
		}
	}

	private void QueryStepTwo()
	{
		for (int i = 0; i < OutputNode.length; i++){
			for(int j = 0; j < HiddenOutput.length; j++) {
				OutputNode [i] = OutputNode [i] + HiddenNode[j] * HiddenOutput[j][i];
			}
		}
		for (int i = 0; i < OutputNode.length; i++) {
			OutputNode[i] = sigmoid(OutputNode[i]);
		}
	}

	private void GetError() {
		for (int i = 0; i < OutputNode.length; i++) {
			//	OutputError[i] = Math.pow(OutputNode[i] - TargetResults[i], 2);
			OutputError[i] = TargetResults[i] - OutputNode[i];
		}

		for (int i = 0; i < HiddenNode.length; i++) {
			HiddenError[i] = 0;
			for (int j = 0; j < OutputNode.length; j++) {
				/*double sum = 0;
				for (int k = 0; k < HiddenNode.length; k++) {
					sum = sum + HiddenOutput[k][j];
				}
				HiddenError[i] = HiddenError[i] + OutputError[j]*HiddenOutput[i][j] / sum;*/
				HiddenError[i] = HiddenError[i] + OutputError[j]*HiddenOutput[i][j];
			}
			//	HiddenError[i] = Math.pow(HiddenError[i], 2);
		}
	}

	private void updateMatrix (double[][] weights, double[] outputs, double[] errors){
		for (int k = 0; k < errors.length; k++) {				//10

			for (int j = 0; j < weights.length; j++) {
				double sum = 0;
				for (int h = 0; h < weights.length; h++) {//784
					sum = sum + weights[h][k] * outputs[h];
				}
				double sig = sigmoid(sum);
				double change = -errors[k] * sig * (1 - sig) * outputs[j];

				weights[j][k] = weights[j][k] - LearningRate * change;
			}

		}

	}

	private static double sigmoid(double x) {
		return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}

	private static void RandomlizeWeights(double[][] weights) {
		Random rng = new Random();
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				weights[i][j] = (rng.nextDouble() * 2 - 1)/ (Math.sqrt(weights.length));
			}
		}

	}

	private void TrainCase(String string) {
		List<Double> result = StringToArray(string);
		double target = result.get(0);
		TargetResults = new double[] {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,};
		TargetResults[(int) target] = 0.99;
		RescaleWeights(result);

		for (int i = 0; i < InputNode.length; i++){
			InputNode[i] = result.get(i);
		}

		query();
		GetError();
		updateMatrix(HiddenOutput, HiddenNode, OutputError);
		double[] removeFirst = Arrays.copyOfRange(InputNode, 1, InputNode.length); //getting rid of the 1st element of the input which is the target value.
		updateMatrix(InputHidden, removeFirst, HiddenError);


	}

	private List<Double> StringToArray(String string){
		string = string + ',';
		List<Double> result = new ArrayList<>();
		String temp = "";
		for (int i = 0; i < string.length(); i++) {
			if (string.charAt(i)!=',') {
				temp = temp + string.charAt(i);
			}
			else{
				result.add(Double.parseDouble(temp));
				temp = "";
			}

		}
		return result;

	}

	private void RescaleWeights(List<Double> weights) {
		for (int i = 0; i < InputNode.length; i++){
			double nWeight = InputNode[i] / 255;
			nWeight = nWeight * 0.99;
			nWeight = nWeight + 0.01;
			InputNode[i] = nWeight;
		}

	}

	private void ReadTrainCases (int NumOfTrain) {
		// The name of the file to open.
		String fileName = TrainFileName;

		// This will reference one line at a time
		String line = null;

		try {
			FileReader fileReader =
					new FileReader(fileName);
			BufferedReader bufferedReader =
					new BufferedReader(fileReader);

			//while((line = bufferedReader.readLine()) != null) {
			for (int i = 0; i < NumOfTrain; i++) {
				line = bufferedReader.readLine();
				TrainCases.add(i, line);

			}
			//}
			bufferedReader.close();
			fileReader.close();
		}
		catch(FileNotFoundException ex) {
			System.out.println(
					"Unable to open file '" +
							fileName + "'");
		}
		catch(IOException ex) {
			System.out.println(
					"Error reading file '"
							+ fileName + "'");
		}

	}

	private void ReadTestCases (int NumOfTest) {
		// The name of the file to open.
		String fileName = TestFileName;
		String line = null;
		try {
			FileReader fileReader =
					new FileReader(fileName);
			BufferedReader bufferedReader =
					new BufferedReader(fileReader);
			for (int i = 0; i < NumOfTest; i++) {
				line = bufferedReader.readLine();
				TestCases.add(i, line);

			}
			bufferedReader.close();
			fileReader.close();
		}
		catch(FileNotFoundException ex) {
			System.out.println(
					"Unable to open file '" +
							fileName + "'");
		}
		catch(IOException ex) {
			System.out.println(
					"Error reading file '"
							+ fileName + "'");
		}

	}

	public static void main(String [] args) throws IOException {

		Tester t = new Tester();
		t.initialize(784, 100, 10, 0.3);
		t.train(50000);
		System.out.println("Training finished");
		t.query(5000);


	}



}
