package cz.muni.fi.pv021.visualization;

import processing.core.PApplet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.logging.Logger;

public class Vis extends PApplet {
    //	An array of stripes
    public static final Logger log = Logger.getLogger("LOG");
    MlpVis mlp;
    public static Vis app;

    public static void main(String[] args){
        PApplet.main("cz.muni.fi.pv021.visualization.Vis");
    }

    public void settings(){
        this.app = this;
        size(500,500);
      }


    int [][] inputs;
    int [][] labels;

    public void setup() {
        mlp = new MlpVis(new int[] {2,3,2}, 0.1d);
        log.info(Arrays.deepToString(mlp.weights.get(0)) + Arrays.deepToString(mlp.weights.get(1)));
        inputs = new int[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        labels = new int[][] {{0,1}, {1,0}, {1,0}, {0,1}};

        mlp.weights.set(0, new Double[][] {{0.1740005435138374, 0.0801473866250636}, {0.3270837708332951, -0.10848856950394965}, {0.16117188643209557, 0.28524559364335067}});
        mlp.weights.set(1, new Double[][] {{-0.041032159382172965, -0.07241668329088848, -0.15394665100235375}, {-0.24672326927202676, -0.09858066687762419, 0.06791824844738917}});

        log.info(Arrays.deepToString(mlp.weights.get(0)) + Arrays.deepToString(mlp.weights.get(1)));

    }

    int iteration = 0;

    public void iterate() {
        for (int j = 0; j < 4; j++) {
            int[][] inp = new int[2][1];
            inp[0][0] = inputs[j][0];
            inp[1][0] = inputs[j][1];
            mlp.evaluate(inp);
            int[][] lab = new int[2][1];
            lab[0][0] = labels[j][0];
            lab[1][0] = labels[j][1];
            mlp.layer3BackProp(inp, lab, 4);
        }

        if (Double.isNaN(mlp.activations.get(1)[0][0])) {
            log.info("Error: " + iteration);
            return;
        }
        if (iteration % 100 == 0) {
            log.info(Arrays.deepToString(mlp.activations.get(1)));
        }

        for (int i = 1; i <= mlp.layers; i++) {
            Neuron[][] nArr = mlp.neurons.get(i-1);
            for(int layer = 0; layer< nArr.length; layer++){
                for(int col = 0; col< nArr[layer].length; col++){
                    log.info(mlp.biases.get(i - 1).toString());
                    log.info(mlp.biases.get(i - 1)[layer].toString());
                    log.info(mlp.biases.get(i - 1)[layer][0].toString());
                    //if(mlp.biases.size() <= i-1 && mlp.biases.get(i - 1).length > layer) {
                        nArr[layer][col].bias = mlp.biases.get(i - 1)[layer][0];
                    //}
                }
            }
        }
        mlp.verbosePrint();
        iteration++;
    }

    ArrayList<Neuron> neurons;

    public long timeMillis = 0;
    public void draw() {
        if(timeMillis == 0 || millis()- timeMillis > 500 ){
            timeMillis = millis();
            iterate();
        }
        background(100);

        Iterator<Neuron[][]> it = mlp.neurons.iterator();

        // Checking the next element availability
        while (it.hasNext()) {
            Neuron[][] nArr = it.next();
            for(int x = 0; x< nArr.length; x++){
                for(int y = 0; y< nArr[x].length; y++){
                    Neuron n = nArr[x][y];
                    n.display();
                }
            }
        }
        text("Iteration: " + iteration,30,30);
    }

}
