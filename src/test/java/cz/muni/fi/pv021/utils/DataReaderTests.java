package cz.muni.fi.pv021.utils;

import cz.muni.fi.pv021.TestBase;
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
    public void printerTest() {
        String file = TEST_DIR_OUT + "test_print.csv";
        double[][][] answers =  new double[][][] {{{1,2}, {1, 2, 3}}};
        DataReader.write(file, answers);
    }


}
