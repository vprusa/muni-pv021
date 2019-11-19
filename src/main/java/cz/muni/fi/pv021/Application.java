package cz.muni.fi.pv021;

import cz.muni.fi.pv021.logic.NeuralNetwork;
import cz.muni.fi.pv021.logic.NeuralNetworkFactory;
import cz.muni.fi.pv021.model.Settings;

/**
 * Run with
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 * @author <a href="mailto:34507957+czFIRE@users.noreply.github.com">Patrik Kadlec</a>
 */
public class Application {

    public static void main(String[] args){
        Settings settings = new Settings();
        NeuralNetwork nn = NeuralNetworkFactory.create(settings);
        nn.learn();
        nn.recognize();
    }

}
