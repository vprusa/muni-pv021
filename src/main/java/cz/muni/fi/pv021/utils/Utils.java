package cz.muni.fi.pv021.utils;

import java.util.Random;

/**
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */

public class Utils {
    public static final int dataSetLength = 60000;
    public static final int testSetLength = 10000;
    public static final int iteration = 1000;

    private static Random random;
    private static long seed;

    /**
     *  Sets the seed for random initialization, will probably get replaced
     *  for better method in the future.
      */
    static {
        seed = System.currentTimeMillis();
        random = new Random(seed);
    }

    public static void setSeed(long seed) {
        Utils.seed = seed;
        random = new Random(seed);
    }

    public static long getSeed() {
        return seed;
    }

    public static double uniformDouble() {
        return random.nextDouble() - 0.5;
    }

    public static int uniformInt(int to) {
        if (to <= 0) {
            throw new IllegalArgumentException("argument should be positive");
        }
        return random.nextInt(to);
    }

    public static double[][] randomMat (int rows, int cols) {
        double[][] mat = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = uniformDouble(); //initialise close to zero
            }
        }

        return mat;
    }

    public static double[][] transposeMat (double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] mat = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[j][i] = matrix[i][j];
            }
        }

        return mat;
    }

    public static double[][] transposeMat (int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] mat = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[j][i] = matrix[i][j];
            }
        }

        return mat;
    }

    public static void addMats (double[][] mat1, double[][] mat2, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] + mat2[i][j];
            }
        }

    }

    public static void addConstantToMat (double num, double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = matrix[i][j] + num;
            }
        }

    }

    public static void subtractMats (double[][] mat1, double[][] mat2, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

    }

    public static void subtractMats (double[][] mat1, int[][] mat2, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

    }

    public static double[][] elementWiseMultiplication (double[][] mat1, double[][] mat2) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        if (mat2.length != rows || mat2[0].length != cols) {
            throw new RuntimeException("Invalid dimensions in elementWiseMultiplication");
        }

        double[][] mat = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] * mat2[i][j];
            }
        }

        return mat;
    }

    public static double[][] multiplyMatByConstant (double num, double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] mat = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = matrix[i][j] * num;
            }
        }
        return mat;
    }

    public static void matrixMultiplication (double[][] mat1, double[][] mat2, double[][] mat) {
        int cols1 = mat1[0].length;
        int rows2 = mat2.length;

        if (cols1 != rows2) {
            throw new RuntimeException("Wrong dimensions for matrix multiplication.");
        }

        int rows1 = mat1.length;
        int cols2 = mat2[0].length;

        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                for (int k = 0; k < cols1; k++) {
                    mat[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }

    }

    public static void matrixMultiplication (double[][] mat1, int[][] mat2, double[][] mat) {
        int cols1 = mat1[0].length;
        int rows2 = mat2.length;

        if (cols1 != rows2) {
            throw new RuntimeException("Wrong dimensions for matrix multiplication.");
        }

        int rows1 = mat1.length;
        int cols2 = mat2[0].length;

        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                for (int k = 0; k < cols1; k++) {
                    mat[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }

    }

    public static void softmax (double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double sum = 0.0d;
        double offset = 0.0d;

        for (double[] matrixR: matrix) {
            for (int i = 0; i < cols; i++) {
                if (offset < matrixR[i]) offset = matrixR[i];
            }
        }

        for (double[] matrixR: matrix) {
            for (int i = 0; i < cols; i++) {
                sum += Math.exp(matrixR[i] - offset);
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = Math.exp(matrix[i][j] - offset) / sum;
            }
        }

    }

    public static void sigmoid (double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = 1/(1 + Math.exp(-matrix[i][j]));
            }
        }

    }

    public static double[][] sigmoidDerivative (double[][] activated) {
        int rows = activated.length;
        int cols = activated[0].length;

        double[][] mat = new double[rows][cols];
        subtractMats(activated, elementWiseMultiplication(activated, activated), mat);
        return mat;
    }

}
