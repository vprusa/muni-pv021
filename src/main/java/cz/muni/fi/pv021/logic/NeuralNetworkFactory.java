package cz.muni.fi.pv021.logic;

import cz.muni.fi.pv021.model.Settings;

/**
 * This is a factory for neural networks
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class NeuralNetworkFactory {

    public static NeuralNetwork create(Settings settings) {
        NeuralNetwork nn = new NeuralNetwork(settings);
        return nn;
    }

}
