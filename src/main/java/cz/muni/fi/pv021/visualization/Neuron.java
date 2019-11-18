package cz.muni.fi.pv021.visualization;

import processing.core.PVector;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Class of Neuron of neural network
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Petr Kadlec</a>
 */
public class Neuron {
    public static final float RADIUS = 50;
    PVector position;
    int x, i;
    Double bias;


    Neuron(int x, int i) {
        this.x = x;
        this.i = i;
        position = new PVector(x*150 + 100, i*150 + 50);
    }

    void display() {
        Vis.app.noStroke();
        Vis.app.fill(204);
        Vis.app.ellipse(position.x, position.y, RADIUS*2, RADIUS*2);
        Vis.app.fill(255,0,255);
        Vis.app.text("B: " + (bias==null ? "" : String.format("%.5f", bias)),position.x-40, position.y+5);
    }
}
