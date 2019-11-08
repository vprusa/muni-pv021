package cz.muni.fi.pv021.utils;

import com.beust.jcommander.Parameter;

/**
 * This class contains settings
 *
 * Settings are set of variables that can be passed via commandline or set as static
 *
 * For documentation check
 *
 * @see <a href="http://jcommander.org/">http://jcommander.org/</a>
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class Settings {

    @Parameter(names = "-n", order = 0)
    public static int iterations;

}
