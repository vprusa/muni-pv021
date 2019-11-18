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

    public static Double uniformDouble() {
        return random.nextDouble() - 0.5;
    }

    public static int uniformInt(int to) {
        if (to <= 0) {
            throw new IllegalArgumentException("argument should be positive");
        }
        return random.nextInt(to);
    }

    public static Double[][] randomMat (int rows, int cols) {
        Double[][] mat = new Double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = uniformDouble(); //initialise close to zero
            }
        }

        return mat;
    }

    public static Double[][] transposeMat (Double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        Double[][] mat = new Double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[j][i] = matrix[i][j];
            }
        }

        return mat;
    }

    public static Double[][] transposeMat (int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        Double[][] mat = new Double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[j][i] = Double.valueOf(matrix[i][j]);
            }
        }

        return mat;
    }

    public static void addMats (Double[][] mat1, Double[][] mat2, Double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] + mat2[i][j];
            }
        }

    }

    public static void addConstantToMat (Double num, Double[][] matrix, Double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = matrix[i][j] + num;
            }
        }

    }

    public static void subtractMats (Double[][] mat1, Double[][] mat2, Double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

    }

    public static void subtractMats (Double[][] mat1, int[][] mat2, Double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

    }

    public static Double[][] elementWiseMultiplication (Double[][] mat1, Double[][] mat2) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        if (mat2.length != rows || mat2[0].length != cols) {
            throw new RuntimeException("Invalid dimensions in elementWiseMultiplication");
        }

        Double[][] mat = new Double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = mat1[i][j] * mat2[i][j];
            }
        }

        return mat;
    }

    public static Double[][] multiplyMatByConstant (Double num, Double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        Double[][] mat = new Double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = matrix[i][j] * num;
            }
        }
        return mat;
    }

    public static void matrixMultiplication (Double[][] mat1, Double[][] mat2, Double[][] mat) {
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

    public static void matrixMultiplication (Double[][] mat1, int[][] mat2, Double[][] mat) {
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

    public static void softmax (Double[][] matrix, Double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        Double sum = 0.0d;
        Double offset = 0.0d;

        for (Double[] matrixR: matrix) {
            for (int i = 0; i < cols; i++) {
                if (offset < matrixR[i]) offset = matrixR[i];
            }
        }

        for (Double[] matrixR: matrix) {
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

    public static void sigmoid (Double[][] matrix, Double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[i][j] = 1/(1 + Math.exp(-matrix[i][j]));
            }
        }

    }

    public static Double[][] sigmoidDerivative (Double[][] activated) {
        int rows = activated.length;
        int cols = activated[0].length;

        Double[][] mat = new Double[rows][cols];
        subtractMats(activated, elementWiseMultiplication(activated, activated), mat);
        return mat;
    }

}
