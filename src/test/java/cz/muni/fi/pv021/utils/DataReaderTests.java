package cz.muni.fi.pv021.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 *
 * This is a test class for {@link DataReader}
 *
 */
public class DataReaderTests {

    /**
     * This is a si a simple silly test
     * */
    @Test
    public void testDataReaderInit(){
        DataReader dr = new DataReader();
        assertNotNull(dr,"DataReader instance is null");
    }

}
