package cz.muni.fi.pv021.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */

public class DataReader {

    public static void readData(BufferedReader data, int[][][] features, int featuresPerLine, int batchSize,
                                 int fileLength) {
        String number;

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
