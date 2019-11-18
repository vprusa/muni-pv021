package cz.muni.fi.pv021;

import cz.muni.fi.pv021.model.Mlp;
import cz.muni.fi.pv021.utils.DataReader;
import cz.muni.fi.pv021.utils.Utils;

import java.util.logging.Logger;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Run with
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Patrik Kadlec</a>
 */
public class Application {
    static final Logger log = Logger.getLogger(Application.class.getSimpleName());

    public static void main(String[] args){
        log.info("Hello, World");
        log.info("Seed: " + Utils.getSeed());
        //new Test();

        Utils.setSeed(138);   //change for better results
        log.info("Seed: " + Utils.getSeed());

        int epochs = 16;
        double learningRate = 0.05;
        double momentum = 0.8;
        int []architecture = new int[] {784, 128, 10};
        int miniBatchSize = 200;
        /* Data provided to us by the teacher */
        String trainData = "D:\\Java\\JavaNeuralNetwork\\MNIST_DATA\\mnist_train_vectors.csv";
        String trainLabels = "D:\\Java\\JavaNeuralNetwork\\MNIST_DATA\\mnist_train_labels.csv";
        String testData = "D:\\Java\\JavaNeuralNetwork\\MNIST_DATA\\mnist_test_vectors.csv";
        String testLabels = "D:\\Java\\JavaNeuralNetwork\\MNIST_DATA\\mnist_test_labels.csv";

        BufferedReader tData = null, tLabels = null, control = null;
        try {
            tData = new BufferedReader(new FileReader(trainData));
            tLabels = new BufferedReader(new FileReader(trainLabels));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert tData != null;
        assert tLabels != null;

        int [][][] batches = new int[Utils.dataSetLength/miniBatchSize][DataReader.dataPerLine][miniBatchSize];
        int [][][] labels = new int[Utils.dataSetLength/miniBatchSize][DataReader.dataClasses][miniBatchSize];

        DataReader.readData(tData, batches, DataReader.dataPerLine, miniBatchSize, Utils.dataSetLength);
        DataReader.readData(tLabels, labels, 1, miniBatchSize, Utils.dataSetLength);

        Mlp mlp = new Mlp(architecture, learningRate, miniBatchSize, momentum);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < Utils.dataSetLength/miniBatchSize; i++) {
                mlp.evaluate(batches[i]);
                mlp.momentumLayer3BackProp(batches[i], labels[i]);
                if (i % 10 == 0) log.info("epoch: " + (epoch + 1) + " iteration: " + i);
            }
        }
        log.info("done learning");

        try {
            control = new BufferedReader(new FileReader(testData));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert control != null;

        int [][][] controlTest = new int[Utils.testSetLength/miniBatchSize][DataReader.dataPerLine][miniBatchSize];
        DataReader.readData(control, controlTest, DataReader.dataPerLine, miniBatchSize, Utils.testSetLength);

        double [][][] answers = new double [Utils.testSetLength/miniBatchSize][miniBatchSize][DataReader.dataClasses];
        for (int i = 0; i < Utils.testSetLength/miniBatchSize; i++) {
            mlp.evaluate(controlTest[i]);
            answers[i] = Utils.transposeMat(mlp.activations.get(1));
        }

        DataReader.write("D:\\Java\\JavaNeuralNetwork\\evaluator\\actualTestPredictions", answers);
    }
}
