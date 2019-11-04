package cz.muni.fi.pv021;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class DataReader {

    final static int dataPerLine = 28 * 28;

    /**
     * Old version of reader, probably removed in next revision
     */
    public static LabelPoint[] readWhole(String data, String dataLabel, int numberOfPoints) throws IOException {
        BufferedReader features = null;
        BufferedReader labels = null;
        String point = "";
        String label = "";
        LabelPoint[] labelPoints = new LabelPoint[numberOfPoints]; //here we can have problems if the last iteration has fever points

        int lineCount = 0;

        try {
            features = new BufferedReader(new FileReader(data));
            labels = new BufferedReader(new FileReader(dataLabel));
            while ((point = features.readLine()) != null && (label = labels.readLine()) != null && lineCount < numberOfPoints) {

                String[] lineValues = point.split(",");
                int[] values = new int[dataPerLine];
                for (int i = 0; i < dataPerLine; i++) {
                    values[i] = Integer.parseInt(lineValues[i]);
                }

                LabelPoint labelPoint = new LabelPoint(Integer.parseInt(label), values);
                labelPoints[lineCount] = labelPoint;
                lineCount++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
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

        System.out.println("Read " + lineCount + " lines.");

        return labelPoints;
    }

    public static LabelPoint[] readBatch(BufferedReader features, BufferedReader labels, int batchSize, LabelPoint[] labelPoints) throws IOException {
        String point = "";
        String label = "";

        int lineCount = 0;

        while ((point = features.readLine()) != null && (label = labels.readLine()) != null && lineCount < batchSize) {
            String[] lineValues = point.split(",");
            int[] values = new int[dataPerLine];
            for (int i = 0; i < dataPerLine; i++) {
                    values[i] = Integer.parseInt(lineValues[i]);
                }

            LabelPoint labelPoint = new LabelPoint(Integer.parseInt(label), values);
            labelPoints[lineCount] = labelPoint;
            lineCount++;
            }

        System.out.println("Read " + lineCount + " csv lines.");

        return labelPoints;
    }

    public static void write(String file, int[] answers) throws FileNotFoundException {
        java.io.File answerCSV = new java.io.File(file);
        java.io.PrintWriter outfile = new java.io.PrintWriter(answerCSV);

        for (int i = 0; i < answers.length; i++) {
            outfile.println(answers[i]);
        }
    }
}
