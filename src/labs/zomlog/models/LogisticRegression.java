package src.labs.zomlog.models;

// SYSTEM IMPORTS
import edu.bu.labs.zomlog.agents.SurvivalAgent;
import edu.bu.labs.zomlog.linalg.Matrix;
import edu.bu.labs.zomlog.linalg.Functions;
import edu.bu.labs.zomlog.utils.Pair;
import edu.bu.labs.zomlog.features.Features.FeatureType;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;

// JAVA PROJECT IMPORTS
import edu.bu.labs.zomlog.agents.SurvivalAgent;
import edu.bu.labs.zomlog.linalg.Matrix;
import edu.bu.labs.zomlog.utils.Pair;

public class LogisticRegression extends Object {

    public static class GradTest extends Object {
        private static final double EPSILON = 1e-4;
        private static final double DELTA = 1e-3;

        public static void checkGrads(LogisticRegression lr,
                                      Matrix X,
                                      Matrix Y_gt,
                                      Pair<Matrix, Matrix> symbolicGradsPair,
                                      double epsilon,
                                      double delta) throws Exception, IndexOutOfBoundsException
        {
            Matrix params[] = new Matrix[]{lr.getW(), lr.getB()};
            Matrix symbolicGrads[] = new Matrix[]{symbolicGradsPair.getFirst(), symbolicGradsPair.getSecond()};
            Matrix numericGrads[] = new Matrix[]{Matrix.zeros_like(lr.getW()), Matrix.zeros_like(lr.getB())};

            BCELoss lf = lr.getLossFunction();

            for (int paramIdx = 0; paramIdx < params.length; ++paramIdx) {
                Matrix P = params[paramIdx];
                Matrix numGrad = numericGrads[paramIdx];

                for (int rIdx = 0; rIdx < numGrad.getShape().getNumRows(); ++rIdx) {
                    for (int cIdx = 0; cIdx < numGrad.getShape().getNumCols(); ++cIdx) {
                        double paramVal = P.get(rIdx, cIdx);

                        double numericalGrad = 0.0;
                        P.set(rIdx, cIdx, paramVal + epsilon);
                        numericalGrad += lf.forward(lr.forward(X), Y_gt).item();

                        P.set(rIdx, cIdx, paramVal - epsilon);
                        numericalGrad -= lf.forward(lr.forward(X), Y_gt).item();

                        numericalGrad /= (2.0 * epsilon);

                        numGrad.set(rIdx, cIdx, numericalGrad);
                        P.set(rIdx, cIdx, paramVal); // reset to original value
                    }
                }
            }

            for (int paramIdx = 0; paramIdx < params.length; ++paramIdx) {
                Matrix P = params[paramIdx];
                Matrix symGrad = symbolicGrads[paramIdx];
                Matrix numGrad = numericGrads[paramIdx];

                double relNorm = symGrad.subtract(numGrad).norm(2).item() /
                                 symGrad.add(numGrad).norm(2).item();
                if (relNorm > delta) {
                    throw new Exception("failed grad check (p.shape=" + P.getShape() +
                                        "): relNorm=" + relNorm + " delta=" + delta +
                                        " epsilon=" + epsilon);
                }
            }
        }
    }

    /**
     * A class that represents the Sigmoid activation Function f(c,x) = c/(1 + e^(-x)).
     */
    public class Sigmoid extends Object {
        private Matrix coeff;

        public Sigmoid() { this(1.0); }

        public Sigmoid(double coeff) {
            this.coeff = Matrix.full(1, 1, coeff);
        }

        private Matrix getCoeff() { return this.coeff; }

        public Matrix forward(Matrix X) throws Exception {
            return this.getCoeff().ediv(Matrix.ones(1, 1).add(
                    Functions.exp(Matrix.full(1, 1, -1.0).emul(X))));
        }

        /**
         * Calculates dLoss_dX = dLoss_dPredictions * (sigmoid'(X)).
         */
        public Matrix backwards(Matrix X, Matrix dLoss_dPredictions) throws Exception {
            Matrix Y_hat = this.forward(X).ediv(this.getCoeff());
            Matrix dPredictions_dX = Y_hat.emul(Matrix.ones(1, 1).subtract(Y_hat))
                                      .emul(this.getCoeff());
            return dLoss_dPredictions.emul(dPredictions_dX);
        }
    }

    public class BCELoss extends Object {
        public BCELoss() {}

        public Matrix forward(Matrix Y_hat, Matrix Y_gt) throws Exception {
            if (!(Y_hat.getShape().equals(Y_gt.getShape()) &&
                  Y_hat.getShape().getNumCols() == 1))
            {
                throw new Exception("[ERROR] LogisticRegression.BCELoss.forward. Y_hat & Y_gt should have same " +
                                    "dimensionality and 1 column each!");
            }

            Matrix YGtNEZeroMask = Y_gt.getRowMaskNEq(0.0, 0);
            Matrix YGtEqZeroMask = Y_gt.getRowMaskEq(0.0, 0);

            double negLogSum1 = -Y_hat.filterRows(YGtNEZeroMask).log().sum().item();
            double negLogSum2 = -Matrix.full(1, 1, 1.0)
                                    .subtract(Y_hat.filterRows(YGtEqZeroMask))
                                    .log().sum().item();
            double loss = (negLogSum1 + negLogSum2) / Y_hat.getShape().getNumRows();
            return Matrix.full(1, 1, loss);
        }

        public Matrix backwards(Matrix Y_hat, Matrix Y_gt) throws Exception {
            if (!(Y_hat.getShape().equals(Y_gt.getShape()) &&
                  Y_hat.getShape().getNumCols() == 1))
            {
                throw new Exception("[ERROR] LogisticRegression.BCELoss.backwards. Y_hat & Y_gt should have same " +
                                    "dimensionality and 1 column each!");
            }

            Matrix dLoss_dY_hat = Matrix.zeros_like(Y_hat);
            for (int rIdx = 0; rIdx < Y_hat.getShape().getNumRows(); ++rIdx) {
                if (Y_gt.get(rIdx, 0) == 1.0) {
                    dLoss_dY_hat.set(rIdx, 0,
                        -1.0 / (Y_hat.get(rIdx, 0) * Y_hat.getShape().getNumRows()));
                } else {
                    dLoss_dY_hat.set(rIdx, 0,
                        1.0 / ((1.0 - Y_hat.get(rIdx, 0)) * Y_hat.getShape().getNumRows()));
                }
            }
            return dLoss_dY_hat;
        }
    }

    // Instance variables: feature headers and model parameters
    private final FeatureType featureHeader[];
    private final Sigmoid       sigmoid;
    private final BCELoss       lossFunction;

    // Parameters of the model (weights and bias)
    private Matrix              w;
    private Matrix              b;

    // Extra credit fields:
    //  - lambda: L2 regularization strength.
    //  - threshold: decision threshold for prediction (tunable using validation data).
    //  - gradCheckEnabled: flag to perform gradient checking (set false for production runs).
    private double              lambda = 0.001;      // L2 regularization hyperparameter (tune as needed)
    private double              threshold = 0.5;     // default decision threshold (will be tuned)
    private boolean             gradCheckEnabled = false; // disable grad checking after debugging

    public LogisticRegression(FeatureType featureHeader[]) {
        this.featureHeader = featureHeader;
        this.w = null;
        this.b = null;
        this.sigmoid = new Sigmoid();
        this.lossFunction = new BCELoss();
    }

    public FeatureType[] getFeatureHeader() { return this.featureHeader; }
    public Matrix getW() { return this.w; }
    public Matrix getB() { return this.b; }
    public final Sigmoid getSigmoid() { return this.sigmoid; }
    public final BCELoss getLossFunction() { return this.lossFunction; }

    private void setW(Matrix m) { this.w = m; }
    private void setB(Matrix m) { this.b = m; }

    // Setter methods for extra credit parameters
    public void setLambda(double lambda) { this.lambda = lambda; }
    public void setGradCheckEnabled(boolean enabled) { this.gradCheckEnabled = enabled; }
    public void setInitialThreshold(double t) { this.threshold = t; }

    /**
     * Forward pass: Compute predictions = sigmoid(XW + b)
     */
    public Matrix forward(Matrix X) throws Exception {
        Matrix Z = X.matmul(this.getW()).add(this.getB());
        return this.getSigmoid().forward(Z);
    }

    /**
     * Backward pass: Compute gradients (dLoss/dW, dLoss/dB) using backpropagation.
     * Also adds an L2 regularization term on the weights.
     *
     * Steps:
     * 1. Compute Z = XW + b.
     * 2. Obtain dLoss_dZ using the sigmoid module's backwards function.
     * 3. Compute gradient with respect to weights: dLoss/dW = X^T * dLoss_dZ.
     * 4. Compute gradient with respect to bias: sum of dLoss_dZ over examples.
     * 5. Add regularization gradient: dLoss/dW += 2 * lambda * W.
     */
    public Pair<Matrix, Matrix> backwards(Matrix X, Matrix dLoss_dPrediction) throws Exception {
        // 1) Compute Z = XW + b
        Matrix Z = X.matmul(this.getW()).add(this.getB());

        // 2) dLoss_dZ = dLoss_dPrediction * sigmoid'(Z)
        Matrix dLoss_dZ = this.getSigmoid().backwards(Z, dLoss_dPrediction);

        // 3) Gradient with respect to weights: dLoss/dW = X^T * dLoss_dZ
        Matrix dLoss_dw = X.transpose().matmul(dLoss_dZ);

        // 4) Gradient with respect to bias: sum of dLoss_dZ over all examples (results in (1,1) Matrix)
        Matrix dLoss_db = dLoss_dZ.sum();

        // 5) Add L2 regularization gradient for weights:
        //    Instead of this.getW().scale(...), we use element-wise multiplication with a constant matrix.
        dLoss_dw = dLoss_dw.add(
            this.getW().emul(
                Matrix.full(
                    this.getW().getShape().getNumRows(),
                    this.getW().getShape().getNumCols(),
                    2.0 * this.lambda
                )
            )
        );

        return new Pair<Matrix, Matrix>(dLoss_dw, dLoss_db);
    }

    /**
     * Fit the model using (optionally decayed) gradient descent.
     *
     * Features for extra credit tuning:
     *  - Adaptive learning rate decay.
     *  - Option to disable gradient checking after initial epoch.
     *  - Number of epochs is parameterized.
     */
    public void fit(Matrix X, Matrix y_gt) {
        int numRows = X.getShape().getNumRows();  // number of examples
        int numCols = X.getShape().getNumCols();    // number of features

        Random rng = new Random(12345);  // ensure repeatability

        // Randomly initialize parameters
        this.setW(Matrix.randn(numCols, 1, rng));
        this.setB(Matrix.randn(1, 1, rng));

        int numEpochs = 50; // increase epochs for better tuning; adjust as necessary
        double initialLearningRate = 1e-3;  // a slightly higher initial LR for faster convergence

        try {
            for (int epochIdx = 0; epochIdx < numEpochs; ++epochIdx) {
                // Apply a simple learning rate decay (for example, inverse decay)
                double learningRate = initialLearningRate / (1.0 + 0.05 * epochIdx);

                Matrix Y_hat = this.forward(X);
                double loss = this.getLossFunction().forward(Y_hat, y_gt).item();

                Pair<Matrix, Matrix> grads = this.backwards(
                                                 X,
                                                 this.getLossFunction().backwards(Y_hat, y_gt)
                                             );

                // Optionally perform gradient checking only in the first epoch
                if (gradCheckEnabled && epochIdx == 0) {
                    GradTest.checkGrads(this, X, y_gt, grads, GradTest.EPSILON, GradTest.DELTA);
                }

                // Update parameters:
                // w = w - learningRate * (dLoss/dw) and b = b - learningRate * (dLoss/db)
                this.setW(this.getW().subtract(
                           Matrix.full(1, 1, learningRate).emul(grads.getFirst())));
                this.setB(this.getB().subtract(
                           Matrix.full(1, 1, learningRate).emul(grads.getSecond())));

                // (Optional) Print training progress periodically
                if ((epochIdx + 1) % 10 == 0) {
                    System.out.println("Epoch " + (epochIdx + 1) + "/" + numEpochs + " loss: " + loss);
                }
            }
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Predicts the class label for a single input example x.
     * Uses the tuned threshold instead of a fixed 0.5.
     *
     * Returns 1 (zombie) if predicted probability >= threshold, else 0 (human).
     */
    public int predict(Matrix x) {
        double prob = 0.0;
        try {
            prob = this.forward(x).item();
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        return (prob >= this.threshold) ? 1 : 0;
    }

    /**
     * Tune the decision threshold using validation data.
     *
     * This method iterates over candidate threshold values (from 0.0 to 1.0)
     * and selects the one that yields zero false positives (i.e. no humans shot)
     * while maximizing true positives (zombies correctly classified).
     * If no threshold yields zero false positives, it selects the threshold with the fewest.
     *
     * @param X_val The validation feature data.
     * @param y_val The ground truth labels for the validation data.
     */
    public void tuneThreshold(Matrix X_val, Matrix y_val) {
        try {
            // Get predictions (raw probabilities) for each validation example.
            Matrix probs = this.forward(X_val);
            int n = probs.getShape().getNumRows();

            double bestThreshold = this.threshold;
            int bestTruePositives = -1;
            int minFalsePositives = Integer.MAX_VALUE;

            // Try candidate thresholds from 0.0 to 1.0 in steps (adjust step size as needed).
            for (double candThreshold = 0.0; candThreshold <= 1.0; candThreshold += 0.01) {
                int falsePositives = 0;
                int truePositives = 0;
                for (int i = 0; i < n; i++) {
                    double p = probs.get(i, 0);
                    int prediction = (p >= candThreshold) ? 1 : 0;
                    int truth = (int) y_val.get(i, 0);
                    if (prediction == 1 && truth == 1) {
                        truePositives++;
                    } else if (prediction == 1 && truth == 0) {
                        falsePositives++;
                    }
                }
                // Prefer candidate thresholds with zero false positives.
                if (falsePositives == 0) {
                    if (truePositives > bestTruePositives) {
                        bestTruePositives = truePositives;
                        bestThreshold = candThreshold;
                    }
                } else {
                    // If none yield zero false positives, choose candidate with fewest false positives.
                    if (falsePositives < minFalsePositives) {
                        minFalsePositives = falsePositives;
                        bestThreshold = candThreshold;
                    }
                }
            }
            // Update the decision threshold.
            this.threshold = bestThreshold;
            System.out.println("Tuned threshold set to: " + bestThreshold);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
