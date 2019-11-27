package cz.muni.fi.pv021.utils;

import cz.muni.fi.pv021.model.Settings;

import java.util.concurrent.ForkJoinPool;
import java.util.function.*;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * This class is used for dealing with stream, mainly for purpose of better performance
 *
 * @author <a href="mailto:prusa.vojtech@email.com">Vojtech Prusa</a>
 */
public class StreamBuilder {

    // better performance by passing parallelStreams here, using finals and reassigning as less as possible
    static boolean parallelStreams = Settings.parallelStreams;
    static boolean useForkJoin = Settings.useForkJoin;
    final static int parallelism = Settings.useForkJoinParallelism;
    // it has to be considered that ForkJoinPool is called sequentially and cannot be nested in itself's calls
    private static ForkJoinPool forkJoinPool = null;

    public static UnaryOperator<IntStream> forkJoinFor(final BiConsumer<IntStream, IntConsumer> streamMethod, final IntConsumer withConsumer) {
        return m -> {
            // TODO remove unnecessary
            //  Notes:
            //  seems like ForkJoin is not the best option here because
            //  - it does not restrict number of used cores
            //  - when used parallelism>50 performance goes down (maybe because of switching context delays?)
            //  - either i misunderstood something crucial or it is just not fitted for this Use Case
            if (useForkJoin && parallelStreams) {
                if (forkJoinPool == null) {
                    forkJoinPool = new ForkJoinPool(parallelism);
                }
                return (IntStream) forkJoinPool.submit(() -> streamMethod.accept(m, withConsumer)).join();
            } else {
                streamMethod.accept(m, withConsumer);
                return m;
            }
        };
    }

    // TODO merge using typed class with new typed inheritance for un-duplication of forkJoinFor ...
    public static UnaryOperator<LongStream> forkJoinFor(final BiConsumer<LongStream, LongConsumer> streamMethod, final LongConsumer withConsumer) {
        return m -> {
            if (useForkJoin && parallelStreams) {
                if (forkJoinPool == null) {
                    forkJoinPool = new ForkJoinPool(parallelism);
                }
                return (LongStream) forkJoinPool.submit(() -> streamMethod.accept(m, withConsumer)).join();
            } else {
                streamMethod.accept(m, withConsumer);
                return m;
            }
        };
    }

    static IntStream rangeInt(final int startInclusive, final int endExclusive) {
        return rangeInt(startInclusive, endExclusive, parallelStreams);
    }

    static IntStream rangeInt(final int startInclusive, final int endExclusive, boolean enableParallelStream) {
        final IntStream is = IntStream.range(startInclusive, endExclusive);
        return (enableParallelStream ? is.parallel() : is);
    }

    static LongStream rangeLong(final int startInclusive, final int endExclusive) {
        return rangeLong(startInclusive, endExclusive, parallelStreams);
    }

    static LongStream rangeLong(final int startInclusive, final int endExclusive, boolean enableParallelStream) {
        final LongStream is = LongStream.range(startInclusive, endExclusive);
        return (enableParallelStream ? is.parallel() : is);
    }

}
