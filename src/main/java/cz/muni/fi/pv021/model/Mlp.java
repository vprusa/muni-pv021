package cz.muni.fi.pv021.model;

import cz.muni.fi.pv021.utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */

public class Mlp {

    static final Logger log = Logger.getLogger(Mlp.class.getSimpleName());

    private double momentum;

    private int layers;
    private int[] architecture;
    private double learningRate;
    private int minibatchSize;

    public List<double[][]> weights;
    private List<double[][]> biases;
    private List<double[][]> potentials;
    public List<double[][]> activations;

    private List<double[][]> diffPotentials;
    private List<double[][]> diffWeights;
    private List<double[][]> diffBiases;

    private List<double[][]> momentumWeights;
    private List<double[][]> momentumBiases;

    public Mlp(int[] architecture, double learningRate, int minibatchSize, double momentum) {
        this.layers = architecture.length - 1;
        this.architecture = architecture;
        this.learningRate = learningRate;
        this.minibatchSize = minibatchSize;
        this.momentum = momentum;

        this.weights = new ArrayList<double[][]>();
        this.biases = new ArrayList<double[][]>();
        this.activations = new ArrayList<double[][]>();
        this.potentials = new ArrayList<double[][]>();

        this.diffBiases = new ArrayList<double[][]>();
        this.diffWeights = new ArrayList<double[][]>();
        this.diffPotentials = new ArrayList<double[][]>();

        this.momentumWeights = new ArrayList<double[][]>();
        this.momentumBiases = new ArrayList<double[][]>();

        for (int i = 1; i <= layers; i++) {
            double[][] mat = Utils.randomMat(architecture[i], architecture[i-1]);
            Utils.multiplyMatByConstant(Math.sqrt(1d/architecture[i-1]), mat);
            this.weights.add(mat);   //can be improved for better initialization
            this.biases.add(new double[architecture[i]][1]);
            this.activations.add(new double[architecture[i]][minibatchSize]);
            this.potentials.add(new double[architecture[i]][minibatchSize]);

            this.diffBiases.add(new double[architecture[i]][1]);
            this.diffPotentials.add(new double[architecture[i]][minibatchSize]);
            this.diffWeights.add(new double[architecture[i]][architecture[i-1]]);

            this.momentumBiases.add(new double[architecture[i]][1]);
            this.momentumWeights.add(new double[architecture[i]][architecture[i-1]]);
        }
    }

    /**
     * Evaluates neural network. Could be much nicer if input was considered as first element of activations
     * @param inputLayer vector of inputs
     */
    public void evaluate(int[][] inputLayer) {
        Utils.matrixMultiplication(weights.get(0), inputLayer, potentials.get(0));
        Utils.addVectorToMat(potentials.get(0), biases.get(0), potentials.get(0));
        Utils.sigmoid(potentials.get(0), activations.get(0));

        if (layers == 1) return;

        for (int i = 1; i < layers - 1; i++) {
            Utils.matrixMultiplication(weights.get(i), activations.get(i - 1), potentials.get(i));  //inner potential
            Utils.addVectorToMat(potentials.get(i), biases.get(i), potentials.get(i));              //add bias
            Utils.sigmoid(potentials.get(i), activations.get(i));                                   //activation func.
        }

        // alone because of different activation function
        Utils.matrixMultiplication(weights.get(layers - 1), activations.get(layers - 2), potentials.get(layers - 1));
        Utils.addVectorToMat(potentials.get(layers - 1), biases.get(layers - 1), potentials.get(layers - 1));
        Utils.softmax(potentials.get(layers - 1), activations.get(layers - 1));
    }

    /**
     * Calculates minibatch gradient descent with momentum
     * @param inputLayer minibatch with inputs
     * @param label minibatch with expected outputs
     */
    public void momentumLayer3BackProp(int[][] inputLayer, int[][] label) {
        Utils.subtractMats(activations.get(layers - 1), label, diffPotentials.get(layers - 1));

        Utils.matrixMultiplication(diffPotentials.get(layers - 1), Utils.transposeMat(activations.get(layers - 2)),
                diffWeights.get(layers - 1));
        Utils.multiplyMatByConstant(1d/minibatchSize, diffWeights.get(layers - 1));

        Utils.meanColumn(diffPotentials.get(layers - 1), diffBiases.get(layers - 1));

        Utils.matrixMultiplication(Utils.transposeMat(weights.get(layers-1)), diffPotentials.get(layers - 1),
                diffPotentials.get(layers - 2));
        Utils.elementWiseMultiplication(diffPotentials.get(layers - 2),
                Utils.sigmoidDerivative(activations.get(layers-2)));

        Utils.matrixMultiplication(diffPotentials.get(layers - 2), Utils.transposeMat(inputLayer),
                diffWeights.get(layers - 2));
        Utils.multiplyMatByConstant(1d/minibatchSize, diffWeights.get(layers - 2));

        Utils.meanColumn(diffPotentials.get(layers - 2), diffBiases.get(layers - 2));

        //momentum
        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumWeights.get(layers - 1)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffWeights.get(layers - 1)),
                momentumWeights.get(layers - 1));
        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumBiases.get(layers - 1)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffBiases.get(layers - 1)),
                momentumBiases.get(layers - 1));

        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumWeights.get(layers - 2)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffWeights.get(layers - 2)),
                momentumWeights.get(layers - 2));
        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumBiases.get(layers - 2)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffBiases.get(layers - 2)),
                momentumBiases.get(layers - 2));

        //update weights and biases
        Utils.subtractMats(weights.get(layers - 1),
                Utils.multiplyMatByConstantNew(learningRate, momentumWeights.get(layers - 1)), weights.get(layers - 1));
        Utils.subtractMats(biases.get(layers - 1),
                Utils.multiplyMatByConstantNew(learningRate, momentumBiases.get(layers - 1)), biases.get(layers - 1));

        Utils.subtractMats(weights.get(layers - 2),
                Utils.multiplyMatByConstantNew(learningRate, momentumWeights.get(layers - 2)), weights.get(layers - 2));
        Utils.subtractMats(biases.get(layers - 2),
                Utils.multiplyMatByConstantNew(learningRate, momentumBiases.get(layers - 2)), biases.get(layers - 2));
    }

    /**
     * Computes minibatch gradient descent
     * @param inputLayer minibatch with inputs
     * @param label minibatch with expected outputs
     */
    public void layer3BackProp(int[][] inputLayer, int[][] label) {
        double[][] dPotentials2 = new double[architecture[layers]][minibatchSize]; //edited
        Utils.subtractMats(activations.get(layers - 1), label, dPotentials2);

        double[][] dWeights2 = new double[architecture[layers]][architecture[layers-1]];
        Utils.matrixMultiplication(dPotentials2, Utils.transposeMat(activations.get(layers - 2)), dWeights2);
        dWeights2 = Utils.multiplyMatByConstantNew(1d/minibatchSize, dWeights2);

        double[][] dBiases2 = new double[architecture[layers]][1];
        Utils.meanColumn(dPotentials2, dBiases2);

        double[][] dPotentials1 = new double[architecture[layers-1]][minibatchSize];    //edited second arg
        Utils.matrixMultiplication(Utils.transposeMat(weights.get(layers-1)), dPotentials2, dPotentials1);
        Utils.elementWiseMultiplication(dPotentials1, Utils.sigmoidDerivative(activations.get(layers-2)));

        double[][] dWeights1 = new double[architecture[layers-1]][architecture[layers-2]];
        Utils.matrixMultiplication(dPotentials1, Utils.transposeMat(inputLayer), dWeights1);
        dWeights1 = Utils.multiplyMatByConstantNew(1d/minibatchSize, dWeights1);

        double[][] dBiases1 = new double[architecture[layers - 1]][1];
        Utils.meanColumn(dPotentials1, dBiases1);

        //update weights and biases
        Utils.subtractMats(weights.get(layers - 2),
                Utils.multiplyMatByConstantNew(learningRate, dWeights1), weights.get(layers - 2));
        Utils.subtractMats(biases.get(layers - 2),
                Utils.multiplyMatByConstantNew(learningRate, dBiases1), biases.get(layers - 2));

        Utils.subtractMats(weights.get(layers - 1),
                Utils.multiplyMatByConstantNew(learningRate, dWeights2), weights.get(layers - 1));
        Utils.subtractMats(biases.get(layers - 1),
                Utils.multiplyMatByConstantNew(learningRate, dBiases2), biases.get(layers - 1));
    }

    public void backPropagate(double[][] inputLayer, int[][] label, int batchSize) {

    }

    public void verbosePrint() {
        for (int i = 0; i < layers; i++) {
            log.info("Layer " + (i+1) + ":");
            log.info("Weights: " + Arrays.deepToString(weights.get(i)));
            log.info("Biases: " + Arrays.deepToString(biases.get(i)));
            log.info("Potentials: " + Arrays.deepToString(potentials.get(i)));
            log.info("Activations: " + Arrays.deepToString(activations.get(i)));
        }
    }

}
