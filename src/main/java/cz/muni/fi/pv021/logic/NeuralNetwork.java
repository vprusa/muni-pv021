package cz.muni.fi.pv021.logic;

import cz.muni.fi.pv021.utils.DataReader;
import cz.muni.fi.pv021.model.Settings;
import cz.muni.fi.pv021.utils.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Logger;

/**
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */
public class NeuralNetwork {

    static final Logger log = Logger.getLogger(NeuralNetwork.class.getSimpleName());

    private final Settings settings;
    private BufferedReader tData, tLabels, control = null;
    private MLP mlp;

    NeuralNetwork(Settings settings){
        this.settings = settings;
        Utils.setSeed(settings.seed);   //change for better results
        log.info("Seed: " + Utils.getSeed());

        this.mlp = new MLP(this.settings);
    }

    public void learn() {
        log.info("Start learning");
        try {
            tData = new BufferedReader(new FileReader(settings.trainData));
            tLabels = new BufferedReader(new FileReader(settings.trainLabels));
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        assert tData != null;
        assert tLabels != null;

        int [][][] batches
                = new int[Utils.dataSetLength/settings.miniBatchSize][settings.dataPerLine][settings.miniBatchSize];
        int [][][] labels
                = new int[Utils.dataSetLength/settings.miniBatchSize][settings.dataClasses][settings.miniBatchSize];

        DataReader.readData(tData, batches, settings.dataPerLine, settings.miniBatchSize, Utils.dataSetLength);
        DataReader.readData(tLabels, labels, 1, settings.miniBatchSize, Utils.dataSetLength);

        for (int epoch = 0; epoch < settings.epochs; epoch++) {
            for (int i = 0; i < Utils.dataSetLength/settings.miniBatchSize; i++) {
                mlp.evaluateAndBackProp(batches[i], labels[i]);
                if (i % 10 == 0) log.info("epoch: " + (epoch + 1) + " iteration: " + i);
            }
        }
        log.info("Done learning");
    }

    public void recognize() {
        log.info("Start recognizing");
        try {
            control = new BufferedReader(new FileReader(settings.testData));
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert control != null;

        int [][][] controlTest
                = new int[Utils.testSetLength/settings.miniBatchSize][settings.dataPerLine][settings.miniBatchSize];
        DataReader.readData(control, controlTest, settings.dataPerLine, settings.miniBatchSize, Utils.testSetLength);

        double [][][] answers = new double
                [Utils.testSetLength/settings.miniBatchSize][settings.miniBatchSize][settings.dataClasses];
        for (int i = 0; i < Utils.testSetLength/settings.miniBatchSize; i++) {
            mlp.evaluate(controlTest[i]);
            answers[i] = Utils.transposeMat(mlp.activations.get(1));
        }

        log.info("Done recognizing and saving answers");
        DataReader.write(settings.answers, answers);
    }
}
