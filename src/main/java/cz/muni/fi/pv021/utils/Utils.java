package cz.muni.fi.pv021.utils;

import java.util.Random;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

/**
 * This class contains
 * - wrappers for generation of numbers
 * - matrix operations
 *
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class Utils {

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

    public static double[][] randomMat(int rows, int cols) {
        double[][] mat = new double[rows][cols];
        StreamBuilder.rangeInt(0, rows).forEach(i ->
                StreamBuilder.rangeInt(0, cols).forEach(j -> {
                    mat[i][j] = uniformDouble();
                    ;
                }));

        return mat;
    }

    public static double[][] transposeMat(double[][] matrix) {
        return transposeDouble().apply(matrix);
        /*
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] mat = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[j][i] = matrix[i][j];
            }
        }
        return mat;
        */
    }

    public static UnaryOperator<double[][]> transposeDouble() {
        return m -> {
            return StreamBuilder.rangeLong(0, m[0].length).mapToObj(r ->
                    StreamBuilder.rangeLong(0, m.length).mapToDouble(c -> m[(int) c][(int) r]).toArray()
            ).toArray(double[][]::new);
        };
    }


    public static int[][] transposeMat(int[][] matrix) {
        return transposeInt().apply(matrix);
        /*
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] mat = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat[j][i] = matrix[i][j];
            }
        }
        return mat;
        */
    }

    public static UnaryOperator<int[][]> transposeInt() {
        return m -> {
            return StreamBuilder.rangeInt(0, m[0].length).mapToObj(r ->
                    StreamBuilder.rangeInt(0, m.length).map(c -> m[c][r]).toArray()
            ).toArray(int[][]::new);
        };
    }

    public static void addMats(double[][] mat1, double[][] mat2, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i ->
                StreamBuilder.rangeInt(0, cols).forEach(j -> {
                    mat[i][j] = mat1[i][j] + mat2[i][j];
                }));
    }

    public static void addConstantToMat(double num, double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i ->
                StreamBuilder.rangeInt(0, cols).forEach(j -> {
                    mat[i][j] = matrix[i][j] + num;
                }));
    }

    public static void meanColumn(double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            mat[i][0] = 0;
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat[i][0] += matrix[i][j];
            });
            mat[i][0] /= cols;
        });
    }

    public static void addVectorToMat(double[][] mat1, double[][] vec, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;


        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat[i][j] = mat1[i][j] + vec[i][0];
            });
        });
    }

    public static void subtractMats(double[][] mat1, double[][] mat2, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            });
        });
    }

    public static void subtractMats(double[][] mat1, int[][] mat2, double[][] mat) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat[i][j] = mat1[i][j] - mat2[i][j];
            });
        });
    }

    public static double[][] elementWiseMultiplicationNew(double[][] mat1, double[][] mat2) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        if (mat2.length != rows || mat2[0].length != cols) {
            throw new RuntimeException("Invalid dimensions in elementWiseMultiplication");
        }

        double[][] mat = new double[rows][cols];
        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat[i][j] = mat1[i][j] * mat2[i][j];
            });
        });
        return mat;
    }

    public static void elementWiseMultiplication(double[][] mat1, double[][] mat2) {
        int rows = mat1.length;
        int cols = mat1[0].length;

        if (mat2.length != rows || mat2[0].length != cols) {
            throw new RuntimeException("Invalid dimensions in elementWiseMultiplication");
        }

        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat1[i][j] *= mat2[i][j];
            });
        });
    }

    public static double[][] multiplyMatByConstantNew(double num, double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] mat = new double[rows][cols];

        StreamBuilder.rangeInt(0, rows).forEach(i -> {
            StreamBuilder.rangeInt(0, cols).forEach(j -> {
                mat[i][j] = matrix[i][j] * num;
            });
        });
        return mat;
    }

    public static void multiplyMatByConstant(double num, double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i ->
                StreamBuilder.rangeInt(0, cols).forEach(j -> {
                    matrix[i][j] *= num;
                }));
    }

    public static void matrixMultiplication(double[][] mat1, double[][] mat2, double[][] mat) {
        int cols1 = mat1[0].length;
        int rows2 = mat2.length;

        if (cols1 != rows2) {
            throw new RuntimeException("Wrong dimensions for matrix multiplication.");
        }

        int rows1 = mat1.length;
        int cols2 = mat2[0].length;
        StreamBuilder.forkJoinFor(IntStream::forEach, i -> {
                    StreamBuilder.rangeInt(0, cols2).forEach(j -> {
                        mat[i][j] = 0;
                        // this cannot be initialized via builder because it has to be sequential so waiting/deadlock
                        // would not occur
                        IntStream.range(0, cols1).forEach(k -> {
                            mat[i][j] += (mat1[i][k] * mat2[k][j]);
                        });
                    });
                }
        ).apply(StreamBuilder.rangeInt(0, rows1));
        /*
        StreamBuilder.rangeInt(0, rows1).forEach(i ->
            StreamBuilder.rangeInt(0, cols2).forEach(j -> {
                mat[i][j] = 0;
                // this cannot be initialized via builder because it has to be sequential so waiting/deadlock
                // would not occur
                IntStream.range(0, cols1).forEach(k -> {
                    mat[i][j] += (mat1[i][k] * mat2[k][j]);
                });
            }));
         */
    }

    public static void matrixMultiplication(double[][] mat1, int[][] mat2, double[][] mat) {
        int cols1 = mat1[0].length;
        int rows2 = mat2.length;

        if (cols1 != rows2) {
            throw new RuntimeException("Wrong dimensions for matrix multiplication.");
        }

        int rows1 = mat1.length;
        int cols2 = mat2[0].length;

        StreamBuilder.rangeInt(0, rows1).forEach(i ->
                StreamBuilder.rangeInt(0, cols2).forEach(j -> {
                    mat[i][j] = 0;
                    IntStream.range(0, cols1).forEach(k -> {
                                mat[i][j] += (mat1[i][k] * mat2[k][j]);
                            }
                    );
                }));
    }

    // TODO use Streams (StreamBuilder)
    public static void softmax(double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double sum;
        double offset;

        for (int j = 0; j < cols; j++) {
            sum = 0.0d;
            offset = 0.0d;
            for (int i = 0; i < rows; i++) {
                if (offset < Math.abs(matrix[i][j])) offset = matrix[i][j];
            }

            for (int i = 0; i < rows; i++) {
                sum += Math.exp(matrix[i][j] - offset);
            }

            for (int i = 0; i < rows; i++) {
                //if (Double.isInfinite(Math.exp(matrix[i][j] - offset)) && Double.isInfinite(sum)) {
                //    mat[i][j] = 1;
                //    continue;
                //}
                mat[i][j] = Math.exp(matrix[i][j] - offset) / sum;
                if (Double.isNaN(mat[i][j])) mat[i][j] = 1;
            }
        }
    }

    public static void sigmoid(double[][] matrix, double[][] mat) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        StreamBuilder.rangeInt(0, rows).forEach(i ->
                StreamBuilder.rangeInt(0, cols).forEach(j -> {
                    mat[i][j] = 1 / (1 + Math.exp(-matrix[i][j]));
                }));
    }

    public static double[][] sigmoidDerivative(double[][] activated) {
        int rows = activated.length;
        int cols = activated[0].length;

        double[][] mat = new double[rows][cols];
        subtractMats(activated, elementWiseMultiplicationNew(activated, activated), mat);
        return mat;
    }
}
