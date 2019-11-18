package cz.muni.fi.pv021.utils;

import cz.muni.fi.pv021.TestBase;
import cz.muni.fi.pv021.model.LabelPoint;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 *
 * This is a test class for {@link DataReader}
 *
 */
public class DataReaderTests extends TestBase {

    /**
     * This is a si a simple silly test
     * */
    @Test
    public void testDataReaderInit(){
        DataReader dr = new DataReader();
        assertNotNull(dr,"DataReader instance is null");
    }

    @Test
    public void readerTest() throws IOException {
        String data = TEST_DIR_RESOURCES + "dataTest.txt";
        String labels = TEST_DIR_RESOURCES + "labelTest.txt";

        ClassLoader classLoader = getClass().getClassLoader();
        File dataFile = new File(classLoader.getResource(data).getFile());
        File labelsFile = new File(classLoader.getResource(labels).getFile());

        BufferedReader features = new BufferedReader(new FileReader(dataFile));
        BufferedReader label = new BufferedReader(new FileReader(labelsFile));

        LabelPoint[] labelPoints = new LabelPoint[2];

        for (int i = 0; i < 2; i++) {
            DataReader.readBatch(features, label, 2, labelPoints);
            for (int j = 0; j < 2; j++) {
                log.info(labelPoints[j].toString());
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

    @Test
    public void printerTest() {
        String file = TEST_DIR_OUT + "test_print.csv";
        double[][][] answers =  new double[][][] {{{1,2}, {1, 2, 3}}};
        DataReader.write(file, answers);
    }


}
