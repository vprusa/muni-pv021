# muni-pv021

1. Your implementation must be compilable and runnable on the Aisa server.
2. The project must contain a runnable script called "RUN" which compiles,
executes and exports everything on "one-click".
3. The implementation must export vectors of train and test predictions.
3a. Exported predictions must be in the same format as is
"actualPredictionsExample" file - on each line is only one number present.
Such number on i-th line represents predicted class index (there are classes
0 - 9 for MNIST) for i-th input vector, hence prediction order is relevant.
3b. Name the exported files "trainPredictions" and "actualTestPredictions".
4. The implementation will take both train and test input vectors, but it must
not touch test data except the evaluation of the already trained model.
4a. If any implementation will touch given test data before the evaluation
of the already trained model, it will be automatically marked as a failed
project.
4b. Why is that - an optimal scenario would hide for you any test data, but
in that case, you would have to deal with serialization and deserialization of
your implementations or you would have to be bound to some given interface and
this is just not wanted in this course.
4c. Don't cheat, your implementations will be checked.
5. It's demanded to reach at least 95% of correct test predictions
(overall accuracy) with given at most half an hour of training time.
5a. Implementations will be executed for a little longer, let's say for 35
minutes. At that time, it should be able to load the data, process them,
train the model and export train/test predictions out to files.
6. The correctness will be checked using an independent program which will be
also
provided for your own purposes.
7. The use of high-level libraries is forbidden. In fact, you don't need any.
7a. Of course, you can use low-level libraries. You definitely can use basic
math functions like exp, sqrt, log, rand, etc. High-level libraries are such
libraries containing matrix-based operations, neural network tools such as
already implemented layers with activation functions, automatic differentiation,
equation/linear-program solvers, etc.
8. What you do internally with the training dataset is up to you.
9. Pack all data with your implementations and put them on the right path so
your program will load them correctly on AISA (project dir. is fine).
