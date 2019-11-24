package cz.muni.fi.pv021.logic;

import cz.muni.fi.pv021.utils.DataReader;
import cz.muni.fi.pv021.model.Settings;
import cz.muni.fi.pv021.utils.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.stream.Stream;

/**
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */
public class NeuralNetwork {

    static final Logger log = Logger.getLogger(NeuralNetwork.class.getSimpleName());
    private Settings settings;
    private MLP mlp;
    private int[][][] batches;
    private int[][][] labels;
    private int[][][] controlTest;
    private double[][][] testLabelsCheck;
    private double[][][] answers;

    private BufferedReader tData = null;
    private BufferedReader tLabels = null;
    private BufferedReader control = null;

    NeuralNetwork(Settings settings) {
        this.settings = settings;
        Utils.setSeed(138);   //change for better results
        log.info("Seed: " + Utils.getSeed());
        try {
            tData = new BufferedReader(new FileReader(settings.trainData));
            tLabels = new BufferedReader(new FileReader(settings.trainLabels));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert tData != null;
        assert tLabels != null;

        batches = new int[Utils.dataSetLength / settings.miniBatchSize][settings.dataPerLine][settings.miniBatchSize];
        labels = new int[Utils.dataSetLength / settings.miniBatchSize][settings.dataClasses][settings.miniBatchSize];

        DataReader.readData(tData, batches, settings.dataPerLine, settings.miniBatchSize, Utils.dataSetLength);
        DataReader.readData(tLabels, labels, 1, settings.miniBatchSize, Utils.dataSetLength);

        mlp = new MLP(settings);
    }

    public void learning() {
        long currentTime = System.currentTimeMillis();
        long prevTime = System.currentTimeMillis();
        long diffTime = 0;
        for (int epoch = 0; epoch < settings.epochs; epoch++) {
            for (int i = 0; i < Utils.dataSetLength / settings.miniBatchSize; i++) {
                //mlp.evaluate(batches[i]);
                mlp.evaluate(batches[i]);
                mlp.momentumLayer3BackProp(batches[i], labels[i]);
                if (i % 10 == 0) {
                    log.info("epoch: " + (epoch + 1) + " iteration: " + i);
                    currentTime = System.currentTimeMillis();
                    diffTime = currentTime - prevTime;
                    log.info("Diff Time: " + diffTime);
                    prevTime = currentTime;
                }
            }
            mlp.setLearningRate(settings.learningRate / Math.sqrt(epoch + 1));
        }
        log.info("Done learning");
    }

    public void evaluations() {
        log.info("Starting evaluations");

        testLabelsCheck = new double[Utils.dataSetLength / settings.miniBatchSize][][];
        for (int i = 0; i < Utils.dataSetLength / settings.miniBatchSize; i++) {
            mlp.evaluate(batches[i]);
            testLabelsCheck[i] = Utils.transposeMat(mlp.getActivations().get(1));
        }

        DataReader.write(settings.trainPredictions, testLabelsCheck);

        log.info("Finished evaluating train set.");
        try {
            control = new BufferedReader(new FileReader(settings.testData));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert control != null;

        controlTest = new int[Utils.testSetLength / settings.miniBatchSize][settings.dataPerLine][settings.miniBatchSize];
        DataReader.readData(control, controlTest, settings.dataPerLine, settings.miniBatchSize, Utils.testSetLength);

        answers = new double[Utils.testSetLength / settings.miniBatchSize][][];
        for (int i = 0; i < Utils.testSetLength / settings.miniBatchSize; i++) {
            mlp.evaluate(controlTest[i]);
            answers[i] = Utils.transposeMat(mlp.getActivations().get(1));
        }

        DataReader.write(settings.answers, answers);
        log.info("Finished evaluating test set.");
    }

}
