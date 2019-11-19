package cz.muni.fi.pv021.model;

import com.beust.jcommander.Parameter;

/**
 * This class contains settings
 *
 * Settings are set of variables that can be passed via commandline or set as static
 *
 * For documentation check
 *
 * @see <a href="http://jcommander.org/">http://jcommander.org/</a>
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class Settings {

    @Parameter(names = "-architecture")
    public static int[] architecture = new int[] {784, 128, 10};;

    public static int layers(){
        return architecture != null ? architecture.length-1 : 0;
    }

    @Parameter(names = "-learningRate")
    public static double learningRate = 0.05;

    @Parameter(names = "-minBatchSize")
    public static int miniBatchSize = 200;

    @Parameter(names = "-momentum")
    public static double momentum = 0.8;

    @Parameter(names = "-epochs")
    public static int epochs = 16;

    //public static String resourcesDir = "D:\\Java\\JavaNeuralNetwork\\MNIST_DATA\\mnist_train_vectors.csv";
    @Parameter(names = "-resourcesDir")
    public static String resourcesDir = "./MNIST_DATA";

    @Parameter(names = "-answers")
    public static String answers = "evaluator/actualTestPredictions";

    /* Data provided to us by the teacher */
    @Parameter(names = "-trainData")
    public static String trainData = "mnist_train_vectors.csv";

    @Parameter(names = "-trainLabels")
    public static String trainLabels = "mnist_train_labels.csv";

    @Parameter(names = "-testData")
    public static String testData = "mnist_test_vectors.csv";

    @Parameter(names = "-testLabels")
    public static String testLabels = "mnist_test_labels.csv";

    @Parameter(names = "-seed")
    public static int seed = 138;

    @Parameter(names = "-dataPerLine")
    public static int dataPerLine = 28 * 28;

    @Parameter(names = "-dataClasses")
    public static int dataClasses = 10;

    public Settings() {}

    public Settings(int[] architecture, double learningRate, int miniBatchSize, double momentum) {
        this.architecture = architecture;
        this.learningRate = learningRate;
        this.miniBatchSize = miniBatchSize;
        this.momentum = momentum;
    }

    /*
    public Settings(int[] architecture, double learningRate, int miniBatchSize, double momentum, int epochs, int seed,
                    String resourcesDir, String answers, String trainData, String trainLabels, String testData,
                    String testLabels, int dataPerLine, int dataClasses) {
        this(architecture, learningRate, miniBatchSize, momentum);
        this.epochs = epochs;
        this.resourcesDir = resourcesDir;
        this.seed = seed;
        this.answers = answers;
        this.trainData = trainData;
        this.trainLabels = trainLabels;
        this.testData = testData;
        this.testLabels = testLabels;
        this.dataPerLine = dataPerLine;
        this.dataClasses = dataClasses;
    }
    */


}
