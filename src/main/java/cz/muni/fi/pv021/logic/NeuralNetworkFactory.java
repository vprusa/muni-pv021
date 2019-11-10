package cz.muni.fi.pv021.logic;

import cz.muni.fi.pv021.utils.Settings;

/**
 * This is a factory for neural networks
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class NeuralNetworkFactory {

    public static NeuralNetwork create(Settings criteria) {
        NeuralNetwork nn = new NeuralNetwork();
        return nn;
    }

}
