package cz.muni.fi.pv021.model;

import java.util.Arrays;

/**
 * 
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Patrik Kadlec</a>
 *
 * */
public class LabelPoint {

    private int label;
    private int[] features;

    public LabelPoint(int label, int[] features) {
        this.label = label;
        this.features = features;
    }

    public int getLabel() {
        return label;
    }

    public int[] getFeatures() {
        return features;
    }

    @Override
    public String toString() {
        return "Labeled point: " + label + " - " + Arrays.toString(features);
    }
}