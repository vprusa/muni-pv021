package cz.muni.fi.pv021.visualization;

import cz.muni.fi.pv021.model.Mlp;
import cz.muni.fi.pv021.utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * This class implements Multilayer perceptron
 *
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */
public class MlpVis {
    static final Logger log = Logger.getLogger(Mlp.class.getSimpleName());

    public int layers;
    public int[] architecture;
    public Double learningRate;

    public List<Double[][]> weights;
    public List<Double[][]> biases;
    public List<Double[][]> potentials;
    public List<Double[][]> activations;

    public List<Double[][]> diffPotentials;
    public List<Double[][]> diffActivations;
    public List<Double[][]> diffWeights;
    public List<Double[][]> diffBiases;

    public List<Neuron[][]> neurons = new ArrayList<Neuron[][]>();

    public Double[][] initMatrix(int cols, int rows){
        Double[][] arr = new Double[cols][rows];
        for(int x = 0; x <cols;x++){
            for(int y = 0; y <rows;y++){
                arr[x][y] = new Double(0.0);
            }
        }
        return arr;
    }

    public MlpVis(int[] architecture, Double learningRate) {
        this.layers = architecture.length - 1;
        this.architecture = architecture;
        this.learningRate = learningRate;

        this.weights = new ArrayList<Double[][]>();
        this.biases = new ArrayList<Double[][]>();
        this.activations = new ArrayList<Double[][]>();
        this.potentials = new ArrayList<Double[][]>();

        for (int i = 1; i <= layers; i++) {
            Double[][] mat = Utils.randomMat(architecture[i], architecture[i-1]);
            mat = Utils.multiplyMatByConstant(Math.sqrt(1d/architecture[i-1]), mat);
            this.weights.add(mat); //can be improved for better initialization
            this.biases.add(initMatrix(architecture[i],1));
            this.activations.add(initMatrix(architecture[i],1));
            this.potentials.add(initMatrix(architecture[i],1));

            Neuron[][] nArr = new Neuron[architecture[i]][1];
            for(int layer = 0; layer< nArr.length; layer++){
                for(int col = 0; col< nArr[layer].length; col++){
                    Neuron n = new Neuron(layer,i);
                    n.bias = this.biases.get(i-1)[layer][0];
                    nArr[layer][col] = n;
                }
            }
            this.neurons.add(nArr);
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
        Double[][] errorOutLayer = initMatrix(architecture[layers],1); //edited
        Utils.subtractMats(activations.get(layers - 1), label, errorOutLayer);

        Double[][] dWeights2 = initMatrix(architecture[layers],architecture[layers-1]);
        Utils.matrixMultiplication(errorOutLayer, Utils.transposeMat(activations.get(layers - 2)), dWeights2);
        dWeights2 = Utils.multiplyMatByConstant(1d/batchSize, dWeights2);

        Double[][] dBiases2 = Utils.multiplyMatByConstant(1d/batchSize, errorOutLayer);

        Double[][] dActivations1 = initMatrix(architecture[layers-1],1);    //edited second arg
        Utils.matrixMultiplication(Utils.transposeMat(weights.get(layers-1)), errorOutLayer, dActivations1);
        Double[][] dPotentials1 = Utils.elementWiseMultiplication(dActivations1,
                Utils.sigmoidDerivative(activations.get(layers-2)));

        Double[][] dWeights1 = initMatrix(architecture[layers-1],architecture[layers-2]);
        Utils.matrixMultiplication(dPotentials1, Utils.transposeMat(inputLayer), dWeights1);
        dWeights1 = Utils.multiplyMatByConstant(1d/batchSize, dWeights1);

        Double[][] dBiases1 = Utils.multiplyMatByConstant(1d/batchSize, dPotentials1);

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

    public void backPropagate(Double[][] inputLayer, int[][] label, int batchSize) {

    }

    public void verbosePrint() {
        String msg = "";
        for (int i = 0; i < layers; i++) {
            msg += "Layer " + (i+1) + ":" + "\n";
            msg += "Weights: " + Arrays.deepToString(weights.get(i)) + "\n";
            msg += "Biases: " + Arrays.deepToString(biases.get(i)) + "\n";
            msg += "Potentials: " + Arrays.deepToString(potentials.get(i)) + "\n";
            msg += "Activations: " + Arrays.deepToString(activations.get(i)) + "\n";
        }
        log.info(msg);
    }

}
