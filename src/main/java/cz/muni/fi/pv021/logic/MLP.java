package cz.muni.fi.pv021.logic;

import cz.muni.fi.pv021.model.Settings;
import cz.muni.fi.pv021.utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */
public class MLP {

    private static final Logger log = Logger.getLogger(MLP.class.getSimpleName());
    private Settings settings;

    private double momentum;
    private double learningRate;

    private List<double[][]> weights;
    private List<double[][]> biases;
    private List<double[][]> potentials;

    private List<double[][]> activations;

    private List<double[][]> diffPotentials;
    private List<double[][]> diffWeights;
    private List<double[][]> diffBiases;

    private List<double[][]> momentumWeights;
    private List<double[][]> momentumBiases;

    public MLP(Settings settings) {
        this.settings = settings;
        this.momentum = settings.momentum;
        this.learningRate = settings.learningRate;

        this.weights = new ArrayList<>();
        this.biases = new ArrayList<>();
        this.activations = new ArrayList<>();
        this.potentials = new ArrayList<>();

        this.diffBiases = new ArrayList<>();
        this.diffWeights = new ArrayList<>();
        this.diffPotentials = new ArrayList<>();

        this.momentumWeights = new ArrayList<>();
        this.momentumBiases = new ArrayList<>();

        for (int i = 1; i <= settings.layers(); i++) {
            double[][] mat = Utils.randomMat(settings.architecture[i], settings.architecture[i - 1]);
            Utils.multiplyMatByConstant(Math.sqrt(1d / settings.architecture[i - 1]), mat);
            this.weights.add(mat);   //can be improved for better initialization
            this.biases.add(new double[settings.architecture[i]][1]);
            this.activations.add(new double[settings.architecture[i]][settings.miniBatchSize]);
            this.potentials.add(new double[settings.architecture[i]][settings.miniBatchSize]);

            this.diffBiases.add(new double[settings.architecture[i]][1]);
            this.diffPotentials.add(new double[settings.architecture[i]][settings.miniBatchSize]);
            this.diffWeights.add(new double[settings.architecture[i]][settings.architecture[i - 1]]);

            this.momentumBiases.add(new double[settings.architecture[i]][1]);
            this.momentumWeights.add(new double[settings.architecture[i]][settings.architecture[i - 1]]);
        }
    }

    /**
     * Evaluates neural network. Could be much nicer if input was considered as first element of activations
     *
     * @param inputLayer vector of inputs
     */
    public void evaluate(int[][] inputLayer) {
        Utils.matrixMultiplication(weights.get(0), inputLayer, potentials.get(0));
        Utils.addVectorToMat(potentials.get(0), biases.get(0), potentials.get(0));
        Utils.sigmoid(potentials.get(0), activations.get(0));

        if (settings.layers() == 1) return;

        for (int i = 1; i < settings.layers() - 1; i++) {
            Utils.matrixMultiplication(weights.get(i), activations.get(i - 1), potentials.get(i));  //inner potential
            Utils.addVectorToMat(potentials.get(i), biases.get(i), potentials.get(i));              //add bias
            Utils.sigmoid(potentials.get(i), activations.get(i));                                   //activation func.
        }

        // alone because of different activation function
        Utils.matrixMultiplication(weights.get(settings.layers() - 1), activations.get(settings.layers() - 2), potentials.get(settings.layers() - 1));
        Utils.addVectorToMat(potentials.get(settings.layers() - 1), biases.get(settings.layers() - 1), potentials.get(settings.layers() - 1));
        Utils.softmax(potentials.get(settings.layers() - 1), activations.get(settings.layers() - 1));
    }

    /**
     * Calculates minibatch gradient descent with momentum
     *
     * @param inputLayer minibatch with inputs
     * @param label      minibatch with expected outputs
     */
    public void momentumLayer3BackProp(int[][] inputLayer, int[][] label) {
        Utils.subtractMats(activations.get(settings.layers() - 1), label, diffPotentials.get(settings.layers() - 1));

        Utils.matrixMultiplication(diffPotentials.get(settings.layers() - 1), Utils.transposeMat(activations.get(settings.layers() - 2)),
                diffWeights.get(settings.layers() - 1));
        Utils.multiplyMatByConstant(1d / settings.miniBatchSize, diffWeights.get(settings.layers() - 1));

        Utils.meanColumn(diffPotentials.get(settings.layers() - 1), diffBiases.get(settings.layers() - 1));

        Utils.matrixMultiplication(Utils.transposeMat(weights.get(settings.layers() - 1)), diffPotentials.get(settings.layers() - 1),
                diffPotentials.get(settings.layers() - 2));
        Utils.elementWiseMultiplication(diffPotentials.get(settings.layers() - 2), Utils.sigmoidDerivative(activations.get(settings.layers() - 2)));

        Utils.matrixMultiplication(diffPotentials.get(settings.layers() - 2), Utils.transposeMat(inputLayer), diffWeights.get(settings.layers() - 2));
        Utils.multiplyMatByConstant(1d / settings.miniBatchSize, diffWeights.get(settings.layers() - 2));

        Utils.meanColumn(diffPotentials.get(settings.layers() - 2), diffBiases.get(settings.layers() - 2));

        //
        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumWeights.get(settings.layers() - 1)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffWeights.get(settings.layers() - 1)),
                momentumWeights.get(settings.layers() - 1));
        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumBiases.get(settings.layers() - 1)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffBiases.get(settings.layers() - 1)),
                momentumBiases.get(settings.layers() - 1));

        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumWeights.get(settings.layers() - 2)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffWeights.get(settings.layers() - 2)),
                momentumWeights.get(settings.layers() - 2));
        Utils.addMats(Utils.multiplyMatByConstantNew(momentum, momentumBiases.get(settings.layers() - 2)),
                Utils.multiplyMatByConstantNew(1 - momentum, diffBiases.get(settings.layers() - 2)),
                momentumBiases.get(settings.layers() - 2));

        //update weights and biases
        Utils.subtractMats(weights.get(settings.layers() - 1),
                Utils.multiplyMatByConstantNew(learningRate, momentumWeights.get(settings.layers() - 1)), weights.get(settings.layers() - 1));
        Utils.subtractMats(biases.get(settings.layers() - 1),
                Utils.multiplyMatByConstantNew(learningRate, momentumBiases.get(settings.layers() - 1)), biases.get(settings.layers() - 1));

        Utils.subtractMats(weights.get(settings.layers() - 2),
                Utils.multiplyMatByConstantNew(learningRate, momentumWeights.get(settings.layers() - 2)), weights.get(settings.layers() - 2));
        Utils.subtractMats(biases.get(settings.layers() - 2),
                Utils.multiplyMatByConstantNew(learningRate, momentumBiases.get(settings.layers() - 2)), biases.get(settings.layers() - 2));
    }

    /**
     * Computes minibatch gradient descent
     *
     * @param inputLayer minibatch with inputs
     * @param label      minibatch with expected outputs
     */
    public void layer3BackProp(int[][] inputLayer, int[][] label) {
        double[][] dPotentials2 = new double[settings.architecture[settings.layers()]][settings.miniBatchSize]; //edited
        Utils.subtractMats(activations.get(settings.layers() - 1), label, dPotentials2);

        double[][] dWeights2 = new double[settings.architecture[settings.layers()]][settings.architecture[settings.layers() - 1]];
        Utils.matrixMultiplication(dPotentials2, Utils.transposeMat(activations.get(settings.layers() - 2)), dWeights2);
        dWeights2 = Utils.multiplyMatByConstantNew(1d / settings.miniBatchSize, dWeights2);

        double[][] dBiases2 = new double[settings.architecture[settings.layers()]][1];
        Utils.meanColumn(dPotentials2, dBiases2);

        double[][] dPotentials1 = new double[settings.architecture[settings.layers() - 1]][settings.miniBatchSize];    //edited second arg
        Utils.matrixMultiplication(Utils.transposeMat(weights.get(settings.layers() - 1)), dPotentials2, dPotentials1);
        Utils.elementWiseMultiplication(dPotentials1, Utils.sigmoidDerivative(activations.get(settings.layers() - 2)));

        double[][] dWeights1 = new double[settings.architecture[settings.layers() - 1]][settings.architecture[settings.layers() - 2]];
        Utils.matrixMultiplication(dPotentials1, Utils.transposeMat(inputLayer), dWeights1);
        dWeights1 = Utils.multiplyMatByConstantNew(1d / settings.miniBatchSize, dWeights1);

        double[][] dBiases1 = new double[settings.architecture[settings.layers() - 1]][1];
        Utils.meanColumn(dPotentials1, dBiases1);

        //update weights and biases
        Utils.subtractMats(weights.get(settings.layers() - 2),
                Utils.multiplyMatByConstantNew(learningRate, dWeights1), weights.get(settings.layers() - 2));
        Utils.subtractMats(biases.get(settings.layers() - 2),
                Utils.multiplyMatByConstantNew(learningRate, dBiases1), biases.get(settings.layers() - 2));

        Utils.subtractMats(weights.get(settings.layers() - 1),
                Utils.multiplyMatByConstantNew(learningRate, dWeights2), weights.get(settings.layers() - 1));
        Utils.subtractMats(biases.get(settings.layers() - 1),
                Utils.multiplyMatByConstantNew(learningRate, dBiases2), biases.get(settings.layers() - 1));
    }

    public void verbosePrint() {
        StringBuilder msg = new StringBuilder();
        for (int i = 0; i < settings.layers(); i++) {
            msg.append("Layer " + (i + 1) + ":\n");
            msg.append("Weights: " + Arrays.deepToString(weights.get(i)) + "\n");
            msg.append("Biases: " + Arrays.deepToString(biases.get(i)) + "\n");
            msg.append("Potentials: " + Arrays.deepToString(potentials.get(i)) + "\n");
            msg.append("Activations: " + Arrays.deepToString(activations.get(i)) + "\n");
        }
        log.info(msg.toString());
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public List<double[][]> getActivations() {
        return activations;
    }

}