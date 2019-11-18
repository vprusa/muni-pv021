package cz.muni.fi.pv021.utils;

import cz.muni.fi.pv021.model.LabelPoint;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */

public class DataReader {

    public static final int dataPerLine = 28 * 28;
    public static final int dataClasses = 10;

    /**
     * Reads batchSize lines from data and labels and puts it into labeled points.
     */
    public static void readBatch(BufferedReader features,BufferedReader labels,int batchSize,LabelPoint[] labelPoints) {
        String point;
        String label;
        int lineCount = 0;

        try {
            while (lineCount<batchSize && (point=features.readLine()) != null && (label=labels.readLine()) != null) {
                String[] lineValues = point.split(",");
                int[][] values = new int[dataPerLine][1];
                for (int i = 0; i < dataPerLine; i++) {
                    values[i][0] = Integer.parseInt(lineValues[i]);
                }
                int[][] lab = new int[1][dataClasses];
                lab[0][Integer.parseInt(label)] = 1;
                LabelPoint labelPoint = new LabelPoint(lab, values);
                labelPoints[lineCount] = labelPoint;
                lineCount++;
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }

    public static void readData (BufferedReader data, int[][][] features, int featuresPerLine, int batchSize,
                                 int fileLength) {
        String number;
        int lineCount;

        try {
            for (int i = 0; i < fileLength/batchSize; i++) {
                for (int cols = 0; cols < batchSize; cols++) {
                    number = data.readLine();
                    if (featuresPerLine != 1) {
                        String[] lineValues;
                        lineValues = number.split(",");
                        for (int rows = 0; rows < featuresPerLine; rows++) {
                            features[i][rows][cols] = Integer.parseInt(lineValues[rows]);
                        }
                    } else {    // takes in one hot
                        features[i][Integer.parseInt(number)][cols] = 1;
                    }
                }
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }

    /**
     * Writes array of ints into file.
     */
    public static void write(String file, double[][][] answers) { //takes in transposed - n/10
        java.io.PrintWriter outfile = null;

        try {
            java.io.File answerCSV = new java.io.File(file);
            outfile = new java.io.PrintWriter(answerCSV);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        assert outfile != null;
        int maxIndex;
        for (double [][] batch : answers) {
            for (double[] answer : batch) {  //change to max
                maxIndex = 0;
                for (int i = 1; i < answer.length; i++) {
                    if (answer[maxIndex] < answer[i]) maxIndex = i;
                }
                outfile.println(maxIndex);
            }
        }
        outfile.close();
    }
}
