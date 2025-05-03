package src.labs.cp;

import java.lang.reflect.Constructor;
import java.util.*;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;

import edu.bu.cp.game.Game;
import edu.bu.cp.linalg.Matrix;
import edu.bu.cp.nn.*;
import edu.bu.cp.nn.layers.*;
import edu.bu.cp.nn.losses.MeanSquaredError;
import edu.bu.cp.nn.models.Sequential;
import edu.bu.cp.nn.optimizers.*;
import edu.bu.cp.utils.*;

public class Main extends Object {
    public static final long SEED = 12345;

    public static Model initQFunction() {
        Sequential m = new Sequential();
        m.add(new Dense(4, 36));
        m.add(new Sigmoid());
        m.add(new Dense(36, 2));
        return m;
    }

    public static int argmax(Matrix qValues) throws IndexOutOfBoundsException {
        Double maxVal = null;
        int action = -1;
        for (int colIdx = 0; colIdx < qValues.getShape().getNumCols(); ++colIdx) {
            double qVal = qValues.get(0, colIdx);
            if (maxVal == null || qVal > maxVal) {
                maxVal = qVal;
                action = colIdx;
            }
        }
        return action;
    }

    public static void train(Game game, Model qFunction, ReplayBuffer rb, Namespace ns) {
        long numTrainingGames = ns.get("numTrainingGames");
        double epsilon = 0.1;
        Random rng = new Random(ns.get("seed"));

        for (int gameIdx = 0; gameIdx < numTrainingGames; ++gameIdx) {
            Matrix state = game.reset();
            boolean isDone = false;

            while (!isDone) {
                int action;
                if (rng.nextDouble() < epsilon) {
                    action = rng.nextInt(2);
                } else {
                    try {
                        Matrix qValues = qFunction.forward(state);
                        action = argmax(qValues);
                    } catch (Exception e) {
                        System.err.println("Error using qFunction in train()");
                        e.printStackTrace();
                        continue;
                    }
                }

                Triple<Matrix, Double, Boolean> obs = game.step(action);
                Matrix nextState = obs.getFirst();
                double reward = obs.getSecond();
                isDone = obs.getThird();

                rb.addSample(state, reward, isDone ? null : nextState);
                state = nextState;
            }
        }
    }

    public static void update(Model qFunction, Optimizer opt, LossFunction lf, ReplayBuffer rb, Random rng, Namespace ns) {
        double gamma = ns.get("gamma");
        int batchSize = ns.get("miniBatchSize");
        int numUpdates = ns.get("numUpdates");

        Pair<Matrix, Matrix> trainingData = rb.getTrainingData(qFunction, gamma);
        Matrix X = trainingData.getFirst();
        Matrix YGt = trainingData.getSecond();

        Dataset ds = new Dataset(X, YGt, batchSize, rng);

        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            ds.shuffle();
            Dataset.BatchIterator it = ds.iterator();
            while (it.hasNext()) {
                Pair<Matrix, Matrix> batch = it.next();
                try {
                    Matrix YHat = qFunction.forward(batch.getFirst());
                    opt.reset();
                    qFunction.backwards(batch.getFirst(), lf.backwards(YHat, batch.getSecond()));
                    opt.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    public static Pair<Double, Double> test(Game game, Model qFunction, Namespace ns) {
        long numEvalGames = ns.get("numEvalGames");
        double gamma = ns.get("gamma");

        double numGames = 0;
        double trajectoryUtilitySum = 0;
        double gameLengthSum = 0;

        for (int gameIdx = 0; gameIdx < numEvalGames; ++gameIdx) {
            double trajectoryUtility = 0;
            Matrix state = game.reset();
            double reward = 0;
            int action = 0;
            boolean isDone = false;
            int t = 0;

            while (!isDone) {
                try {
                    Matrix qValues = qFunction.forward(state);
                    action = argmax(qValues);
                } catch (Exception e) {
                    System.err.println("Main.main: error caught using qFunction");
                    e.printStackTrace();
                }

                Triple<Matrix, Double, Boolean> obs = game.step(action);
                state = obs.getFirst();
                reward = obs.getSecond();
                isDone = obs.getThird();

                trajectoryUtility += Math.pow(gamma, t) * reward;
                t += 1;
            }

            trajectoryUtilitySum += trajectoryUtility;
            gameLengthSum += t;
            numGames += 1;
        }

        return new Pair<>(trajectoryUtilitySum / numGames, gameLengthSum / numGames);
    }

    public static void main(String[] args) {
        ArgumentParser parser = ArgumentParsers.newFor("Main").build()
                .defaultHelp(true)
                .description("Play openai-gym Deterministic Mountain Car in Java");

        parser.addArgument("-p", "--numCycles").type(Long.class).setDefault(1L);
        parser.addArgument("-t", "--numTrainingGames").type(Long.class).setDefault(10L);
        parser.addArgument("-v", "--numEvalGames").type(Long.class).setDefault(5L);
        parser.addArgument("-b", "--maxBufferSize").type(Integer.class).setDefault(1280);
        parser.addArgument("-r", "--replacementType").type(ReplayBuffer.ReplacementType.class).setDefault(ReplayBuffer.ReplacementType.RANDOM);
        parser.addArgument("-u", "--numUpdates").type(Integer.class).setDefault(1);
        parser.addArgument("-m", "--miniBatchSize").type(Integer.class).setDefault(128);
        parser.addArgument("-n", "--lr").type(Double.class).setDefault(1e-6);
        parser.addArgument("-c", "--clip").type(Double.class).setDefault(100d);
        parser.addArgument("-d", "--optimizerType").type(String.class).setDefault("sgd");
        parser.addArgument("-b1", "--beta1").type(Double.class).setDefault(0.9);
        parser.addArgument("-b2", "--beta2").type(Double.class).setDefault(0.999);
        parser.addArgument("-g", "--gamma").type(Double.class).setDefault(1e-4);
        parser.addArgument("-i", "--inFile").type(String.class).setDefault("");
        parser.addArgument("-o", "--outFile").type(String.class).setDefault("./params/qFunction");
        parser.addArgument("--outOffset").type(Long.class).setDefault(0L);
        parser.addArgument("--seed").type(Long.class).setDefault(SEED);

        Namespace ns = parser.parseArgsOrFail(args);

        long numCycles = ns.get("numCycles");
        long seed = ns.get("seed");
        String checkpointFileBase = ns.get("outFile");
        long offset = ns.get("outOffset");

        Random rng = new Random(seed);
        Game game = new Game(rng);
        Model qFunction = initQFunction();
        LossFunction lf = new MeanSquaredError();
        ReplayBuffer rb = new ReplayBuffer(ns.get("replacementType"), ns.get("maxBufferSize"), 4, rng);

        double bestGameLength = 0;

        for (int cycleIdx = 0; cycleIdx < numCycles; ++cycleIdx) {
            double baseLr = ns.get("lr");
            double decayedLr = baseLr * Math.pow(0.9, cycleIdx / 50);
            Optimizer opt = new SGDOptimizer(qFunction.getParameters(), decayedLr);

            train(game, qFunction, rb, ns);
            update(qFunction, opt, lf, rb, rng, ns);

            Pair<Double, Double> expectedUtilityAndAvgGameLength = test(game, qFunction, ns);
            double avgUtil = expectedUtilityAndAvgGameLength.getFirst();
            double avgGameLength = expectedUtilityAndAvgGameLength.getSecond();

            System.out.println("after cycle=" + cycleIdx + " avg(utility)=" + avgUtil + " avg(game_length)=" + avgGameLength);
            qFunction.save(checkpointFileBase + (cycleIdx + offset) + ".model");

            if (avgGameLength > bestGameLength) {
                bestGameLength = avgGameLength;
                qFunction.save("params.model");
                System.out.println("New best model saved with avgGameLength=" + bestGameLength);
            }
        }
    }
}
