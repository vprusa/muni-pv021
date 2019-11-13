package cz.muni.fi.pv021.model;

import cz.muni.fi.pv021.utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */

public class Mlp {

    private int layers;
    private int[] architecture;
    private double learningRate;

    public List<double[][]> weights;
    private List<double[][]> biases;
    private List<double[][]> potentials;
    public List<double[][]> activations;

    private List<double[][]> diffPotentials;
    private List<double[][]> diffActivations;
    private List<double[][]> diffWeights;
    private List<double[][]> diffBiases;

    public Mlp(int[] architecture, double learningRate) {
        this.layers = architecture.length - 1;
        this.architecture = architecture;
        this.learningRate = learningRate;

        this.weights = new ArrayList<double[][]>();
        this.biases = new ArrayList<double[][]>();
        this.activations = new ArrayList<double[][]>();
        this.potentials = new ArrayList<double[][]>();

        for (int i = 1; i <= layers; i++) {
            double[][] mat = Utils.randomMat(architecture[i], architecture[i-1]);
            mat = Utils.multiplyMatByConstant(Math.sqrt(1d/architecture[i-1]), mat);
            this.weights.add(mat);   //can be improved for better initialization
            this.biases.add(new double[architecture[i]][1]);
            this.activations.add(new double[architecture[i]][1]);
            this.potentials.add(new double[architecture[i]][1]);
        }
    }

    /**
     * Evaluates neural network. Could be much nicer if input was considered as first element of activations
     * @param inputLayer vector of inputs
     */
    public void evaluate(int[][] inputLayer) {
        Utils.matrixMultiplication(weights.get(0), inputLayer, potentials.get(0));
        Utils.addMats(potentials.get(0), biases.get(0), potentials.get(0));
        Utils.sigmoid(potentials.get(0), activations.get(0));

        if (layers == 1) return;

        for (int i = 1; i < layers - 1; i++) {
            Utils.matrixMultiplication(weights.get(i), activations.get(i - 1), potentials.get(i));  //inner potential
            Utils.addMats(potentials.get(i), biases.get(i), potentials.get(i));                     //add bias
            Utils.sigmoid(potentials.get(i), activations.get(i));                                   //activation func.
        }

        // alone because of different activation function
        Utils.matrixMultiplication(weights.get(layers - 1), activations.get(layers - 2), potentials.get(layers - 1));
        Utils.addMats(potentials.get(layers - 1), biases.get(layers - 1), potentials.get(layers - 1));
        Utils.softmax(potentials.get(layers - 1), activations.get(layers - 1));
    }

    public void layer3BackProp(int[][] inputLayer, int[][] label, int batchSize) { //label must be a matrix 1*10 matrix
        double[][] errorOutLayer = new double[architecture[layers]][1]; //edited
        Utils.subtractMats(activations.get(layers - 1), label, errorOutLayer);

        double[][] dWeights2 = new double[architecture[layers]][architecture[layers-1]];
        Utils.matrixMultiplication(errorOutLayer, Utils.transposeMat(activations.get(layers - 2)), dWeights2);
        dWeights2 = Utils.multiplyMatByConstant(1d/batchSize, dWeights2);

        double[][] dBiases2 = Utils.multiplyMatByConstant(1d/batchSize, errorOutLayer);

        double[][] dActivations1 = new double[architecture[layers-1]][1];    //edited second arg
        Utils.matrixMultiplication(Utils.transposeMat(weights.get(layers-1)), errorOutLayer, dActivations1);
        double[][] dPotentials1 = Utils.elementWiseMultiplication(dActivations1,
                Utils.sigmoidDerivative(activations.get(layers-2)));

        double[][] dWeights1 = new double[architecture[layers-1]][architecture[layers-2]];
        Utils.matrixMultiplication(dPotentials1, Utils.transposeMat(inputLayer), dWeights1);
        dWeights1 = Utils.multiplyMatByConstant(1d/batchSize, dWeights1);

        double[][] dBiases1 = Utils.multiplyMatByConstant(1d/batchSize, dPotentials1);

        //update weights and biases
        Utils.subtractMats(weights.get(layers - 2),
                Utils.multiplyMatByConstant(learningRate, dWeights1), weights.get(layers - 2));
        Utils.subtractMats(biases.get(layers - 2),
                Utils.multiplyMatByConstant(learningRate, dBiases1), biases.get(layers - 2));

        Utils.subtractMats(weights.get(layers - 1),
                Utils.multiplyMatByConstant(learningRate, dWeights2), weights.get(layers - 1));
        Utils.subtractMats(biases.get(layers - 1),
                Utils.multiplyMatByConstant(learningRate, dBiases2), biases.get(layers - 1));
    }

    public void backPropagate(double[][] inputLayer, int[][] label, int batchSize) {

    }

    public void verbosePrint() {
        for (int i = 0; i < layers; i++) {
            System.out.println("Layer " + (i+1) + ":");
            System.out.println("Weights: " + Arrays.deepToString(weights.get(i)));
            System.out.println("Biases: " + Arrays.deepToString(biases.get(i)));
            System.out.println("Potentials: " + Arrays.deepToString(potentials.get(i)));
            System.out.println("Activations: " + Arrays.deepToString(activations.get(i)));
        }
    }

}
