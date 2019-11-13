package cz.muni.fi.pv021;

import cz.muni.fi.pv021.model.LabelPoint;
import cz.muni.fi.pv021.model.Mlp;
import cz.muni.fi.pv021.utils.DataReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */

class Test {

    Test() {
        //readerTest("D:\\Java\\JavaNeuralNetwork\\testData\\dataTest.txt", "D:\\Java\\JavaNeuralNetwork\\testData\\labelTest.txt");
        //printerTest("D:\\Java\\JavaNeuralNetwork\\testData\\test_print.csv", new int[][] {{1,2}, {1, 2, 3}});
        //XORTest();
        evaluateTest();
    }

    private void readerTest (String data, String labels) throws IOException {
        BufferedReader features = new BufferedReader(new FileReader(data));
        BufferedReader label = new BufferedReader(new FileReader(labels));

        LabelPoint[] labelPoints = new LabelPoint[2];

        for (int i = 0; i < 2; i++) {
            DataReader.readBatch(features, label, 2, labelPoints);
            for (int j = 0; j < 2; j++) {
                System.out.println(labelPoints[j]);
            }
        }

        if (features != null) {
            try {
                features.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (label != null) {
            try {
                features.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void printerTest(String file, int[][] answers) {
        DataReader.write(file, answers);
    }


    //konverguje pro [[0.2050401577706189, -0.27902545333530554], [0.18523162431881537, -0.16910199085845082], [-0.30764552128935074, 0.3475919129379542]][[-0.04818224942573579, -0.19983994177838005, 0.23143088846252757], [0.0074861180571732305, -0.10875984307543428, -0.1620885434265968]]
    //failne na i=685 [[0.1740005435138374, 0.0801473866250636], [0.3270837708332951, -0.10848856950394965], [0.16117188643209557, 0.28524559364335067]][[-0.041032159382172965, -0.07241668329088848, -0.15394665100235375], [-0.24672326927202676, -0.09858066687762419, 0.06791824844738917]]
    private void XORTest() {
        Mlp mlp = new Mlp(new int[] {2,3,2}, 0.1d);
        System.out.println(Arrays.deepToString(mlp.weights.get(0)) + Arrays.deepToString(mlp.weights.get(1)));
        int [][] inputs = new int[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int [][] labels = new int[][] {{0,1}, {1,0}, {1,0}, {0,1}};

        mlp.weights.set(0, new double[][] {{0.1740005435138374, 0.0801473866250636}, {0.3270837708332951, -0.10848856950394965}, {0.16117188643209557, 0.28524559364335067}});
        mlp.weights.set(1, new double[][] {{-0.041032159382172965, -0.07241668329088848, -0.15394665100235375}, {-0.24672326927202676, -0.09858066687762419, 0.06791824844738917}});
        System.out.println(Arrays.deepToString(mlp.weights.get(0)) + Arrays.deepToString(mlp.weights.get(1)));
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 4; j++) {
                int[][] inp = new int[2][1];
                inp[0][0] = inputs[j][0];
                inp[1][0] = inputs[j][1];
                mlp.evaluate(inp);
                int[][] lab = new int[2][1];
                lab[0][0] = labels[j][0];
                lab[1][0] = labels[j][1];
                mlp.layer3BackProp(inp, lab, 4);
            }

            if (Double.isNaN(mlp.activations.get(1)[0][0])) {
                System.out.println("Error: " + i);
                return;
            }
            if (i % 100 == 0) {
                System.out.println(Arrays.deepToString(mlp.activations.get(1)));
            }
        }

    }

    private void evaluateTest() {
        Mlp mlp = new Mlp(new int[] {2,2,2}, 0.1);
        mlp.weights.set(0, new double[][] {{0.1, 0.3}, {0.2, 0.4}});
        mlp.weights.set(1, new double[][] {{0.1, 0.3}, {0.2, 0.4}});
        System.out.println(Arrays.deepToString(mlp.weights.get(0)) + Arrays.deepToString(mlp.weights.get(1)));

        int[][] inp = new int[2][1];
        inp[0][0] = 1;
        inp[1][0] = 2;

        int[][] lab = new int[2][1];
        lab[0][0] = 0;
        lab[1][0] = 1;

        mlp.evaluate(inp);
        mlp.verbosePrint();

        mlp.layer3BackProp(inp, lab, 1);
        mlp.verbosePrint();
    }
}
