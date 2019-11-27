package cz.muni.fi.pv021.model;

import java.util.*;
import java.util.logging.Logger;

/**
 * This class contains settings
 * <p>
 * Settings are set of variables that can be passed via commandline and have default values
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class Settings {

    static final Logger log = Logger.getLogger(Settings.class.getSimpleName());

    public Settings(String[] args) {
        List<String> options = null;
        for (int i = 0; i < args.length; i++) {
            final String a = args[i];
            log.info("Parsing argument: " + a);

            if (a.charAt(0) == '-') {
                if (a.length() < 2) {
                    log.severe("Error at argument " + a);
                    break;
                }
                final String argNameFull = a.substring(0, a.indexOf('=') + 1);
                final String argName = argNameFull.substring(argNameFull.lastIndexOf('-') + 1, argNameFull.indexOf('='));
                options = new ArrayList<>();
                switch (argName) {
                    case "architecture":
                        this.architecture = Arrays.stream(a.replace(argNameFull, "")
                                .split(",")).mapToInt(Integer::parseInt).toArray();
                        break;
                    case "learningRate":
                        this.learningRate = Double.parseDouble(a.replace(argNameFull, ""));
                        break;
                    case "miniBatchSize":
                        this.miniBatchSize = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "momentum":
                        this.momentum = Double.parseDouble(a.replace(argNameFull, ""));
                        break;
                    case "epochs":
                        this.epochs = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "parallelStreams":
                        this.parallelStreams = Boolean.parseBoolean(a.replace(argNameFull, ""));
                        break;
                    case "useForkJoin":
                        this.useForkJoin = Boolean.parseBoolean(a.replace(argNameFull, ""));
                        break;
                    case "useForkJoinParallelism":
                        this.useForkJoinParallelism = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "resourcesDir":
                        this.resourcesDir = a.replace(argNameFull, "");
                        break;
                    case "seed":
                        this.seed = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "trainPredictions":
                        this.trainPredictions = a.replace(argNameFull, "");
                        break;
                    case "answers":
                        this.answers = a.replace(argNameFull, "");
                        break;
                    case "trainData":
                        this.trainData = a.replace(argNameFull, "");
                        break;
                    case "trainLabels":
                        this.trainLabels = a.replace(argNameFull, "");
                        break;
                    case "testData":
                        this.testData = a.replace(argNameFull, "");
                        break;
                    case "testLabels":
                        this.testLabels = a.replace(argNameFull, "");
                        break;
                    case "dataPerLine":
                        this.dataPerLine = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "dataClasses":
                        this.dataClasses = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "dataSetLength":
                        this.dataSetLength = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    case "testSetLength":
                        this.testSetLength = Integer.parseInt(a.replace(argNameFull, ""));
                        break;
                    default:
                        log.severe("Illegal parameter usage");
                }
            } else if (options != null) {
                options.add(a);
            } else {
                log.severe("Illegal parameter usage");
                break;
            }
        }
        printSettings();
    }

    public Settings(int[] architecture, double learningRate, int miniBatchSize, double momentum) {
        this.architecture = architecture;
        this.learningRate = learningRate;
        this.miniBatchSize = miniBatchSize;
        this.momentum = momentum;
    }

    public static int[] architecture = new int[]{784, 128, 10};
    ;

    public int layers() {
        return architecture != null ? architecture.length - 1 : -1;
    }

    public static double learningRate = 0.05;

    public static int miniBatchSize = 200;

    public static double momentum = 0.80;

    public static int epochs = 16;

    public static boolean parallelStreams = false;

    public static boolean useForkJoin = true;

    // consider that the depth of recursion implementation of evaluation parallel streams is 2
    // (ideally use multiplications of 2?)
    // also consider that max number should be number of processors (maybe -1 because of main thread)
    public static int useForkJoinParallelism = 7;

    public static String resourcesDir = "./MNIST_DATA/";

    // results a.k.a answers
    public static String answers = "evaluator/actualTestPredictions";

    // results on train test
    public static String trainPredictions = "evaluator/trainPredictions";

    /* Data provided to us by the teacher */
    public static String trainData = resourcesDir + "mnist_train_vectors.csv";

    public static String trainLabels = resourcesDir + "mnist_train_labels.csv";

    public static String testData = resourcesDir + "mnist_test_vectors.csv";

    public static String testLabels = resourcesDir + "mnist_test_labels.csv";

    public static int seed = 138;

    public static int dataPerLine = 28 * 28;

    public static int dataClasses = 10;

    public static int dataSetLength = 60000;

    public static int testSetLength = 10000;

    public void printSettings() {
        StringBuilder msg = new StringBuilder("");
        msg.append("architecture: " + architecture + "\n");
        msg.append("learningRate: " + learningRate + "\n");
        msg.append("miniBatchSize: " + miniBatchSize + "\n");
        msg.append("momentum: " + momentum + "\n");
        msg.append("epochs: " + epochs + "\n");
        msg.append("parallelStreams: " + parallelStreams + "\n");
        msg.append("useForkJoin: " + useForkJoin + "\n");
        msg.append("useForkJoinParallelism: " + useForkJoinParallelism + "\n");
        msg.append("resourcesDir: " + resourcesDir + "\n");
        msg.append("seed: " + seed + "\n");
        msg.append("answers: " + answers + "\n");
        msg.append("trainData: " + trainData + "\n");
        msg.append("trainLabels: " + trainLabels + "\n");
        msg.append("testData: " + testData + "\n");
        msg.append("testLabels: " + testLabels + "\n");
        msg.append("dataPerLine: " + dataPerLine + "\n");
        msg.append("dataClasses: " + dataClasses + "\n");
        msg.append("dataSetLength: " + dataSetLength + "\n");
        msg.append("testSetLength: " + testSetLength + "\n");
        log.info(msg.toString());
    }

}
