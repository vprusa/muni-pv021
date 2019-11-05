package cz.muni.fi.pv021;

import java.util.Arrays;

public class LabelPoint {

    private int label;
    private int[] features;

    LabelPoint(int label, int[] features) {
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