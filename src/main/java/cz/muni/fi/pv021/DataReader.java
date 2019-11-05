package cz.muni.fi.pv021;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;

class DataReader {

    private final static int dataPerLine = 28 * 28;

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

    /**
     * Tested and functional, reads batchSize lines from data and labels and puts it into labeled points.
     */
    static void readBatch(BufferedReader features, BufferedReader labels, int batchSize, LabelPoint[] labelPoints) {
        String point;
        String label;
        int lineCount = 0;

        try {
            while (lineCount < batchSize && (point = features.readLine()) != null && (label = labels.readLine()) != null) {
                String[] lineValues = point.split(",");
                int[] values = new int[dataPerLine];
                for (int i = 0; i < dataPerLine; i++) {
                    values[i] = Integer.parseInt(lineValues[i]);
                }

                LabelPoint labelPoint = new LabelPoint(Integer.parseInt(label), values);
                labelPoints[lineCount] = labelPoint;
                lineCount++;
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }

    /**
     * Tested and functional, writes array of ints into file.
     */
    static void write(String file, int[] answers) {
        java.io.PrintWriter outfile = null;

        try {
            java.io.File answerCSV = new java.io.File(file);
            outfile = new java.io.PrintWriter(answerCSV);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        assert outfile != null;
        for (int answer : answers) {
            outfile.println(answer);
        }
        outfile.close();
    }
}