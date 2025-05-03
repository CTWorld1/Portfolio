package src.labs.cp;


// SYSTEM IMPORTS
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.cp.linalg.Matrix;
import edu.bu.cp.nn.Model;
import edu.bu.cp.utils.Pair;


public class ReplayBuffer
    extends Object
{

    public static enum ReplacementType
    {
        RANDOM,
        OLDEST;
    }

    private ReplacementType     type;
    private int                 size;
    private int                 newestSampleIdx;

    private Matrix              prevStates;
    private Matrix              rewards;
    private Matrix              nextStates;
    private boolean             isStateTerminalMask[];

    private Random              rng;

    public ReplayBuffer(ReplacementType type,
                        int numSamples,
                        int dim,
                        Random rng)
    {
        this.type = type;
        this.size = 0;
        this.newestSampleIdx = -1;

        this.prevStates = Matrix.zeros(numSamples, dim);
        this.rewards = Matrix.zeros(numSamples, 1);
        this.nextStates = Matrix.zeros(numSamples, dim);
        this.isStateTerminalMask = new boolean[numSamples];

        this.rng = rng;

    }

    public int size() { return this.size; }
    public final ReplacementType getReplacementType() { return this.type; }
    private int getNewestSampleIdx() { return this.newestSampleIdx; }
    private Matrix getPrevStates() { return this.prevStates; }
    private Matrix getNextStates() { return this.nextStates; }
    private Matrix getRewards() { return this.rewards; }
    private boolean[] getIsStateTerminalMask() { return this.isStateTerminalMask; }

    private Random getRandom() { return this.rng; }

    private void setSize(int i) { this.size = i; }
    private void setNewestSampleIdx(int i) { this.newestSampleIdx = i; }

    private int chooseSampleToEvict()
    {
        int idxToEvict = -1;

        switch(this.getReplacementType())
        {
            case RANDOM:
                idxToEvict = this.getRandom().nextInt(this.getNextStates().getShape().getNumRows());
                break;
            case OLDEST:
                idxToEvict = (this.getNewestSampleIdx() + 1) % this.getNextStates().getShape().getNumRows();
                break;
            default:
                System.err.println("[ERROR] ReplayBuffer.chooseSampleToEvict: unknown replacement type "
                    + this.getReplacementType());
                System.exit(-1);
        }

        return idxToEvict;
    }

    public void addSample(Matrix prevState,
                          double reward,
                          Matrix nextState)
    {
        int capacity  = this.getPrevStates().getShape().getNumRows();
        int insertIdx;

        /* -----------------------------------------------------------
         * Decide where this new transition should go
         * ----------------------------------------------------------- */
        if (this.size() < capacity)            // still room ⇒ append
        {
            insertIdx = this.size();
            this.setSize(this.size() + 1);
        }
        else                                   // buffer full ⇒ evict
        {
            insertIdx = chooseSampleToEvict();
        }

        /* -----------------------------------------------------------
         * Copy data into the chosen row
         * ----------------------------------------------------------- */
        try
        {
            //  s
            this.getPrevStates()
                .copySlice(insertIdx, insertIdx + 1,
                           0,
                           this.getPrevStates().getShape().getNumCols(),
                           prevState);

            //  r
            this.getRewards().set(insertIdx, 0, reward);

            //  s′  (store only if non‑terminal)
            if (nextState != null)
            {
                this.getNextStates()
                    .copySlice(insertIdx, insertIdx + 1,
                               0,
                               this.getNextStates().getShape().getNumCols(),
                               nextState);
                this.getIsStateTerminalMask()[insertIdx] = false;
            }
            else    // terminal transition
            {
                this.getIsStateTerminalMask()[insertIdx] = true;
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }

        /* -----------------------------------------------------------
         * Book‑keeping so OLDEST replacement keeps working
         * ----------------------------------------------------------- */
        this.setNewestSampleIdx(insertIdx);
    }


    public static double max(Matrix qValues) throws IndexOutOfBoundsException
    {
        double maxVal = 0;
        boolean initialized = false;

        for(int colIdx = 0; colIdx < qValues.getShape().getNumCols(); ++colIdx)
        {
            double qVal = qValues.get(0, colIdx);
            if(!initialized || qVal > maxVal)
            {
                maxVal = qVal;
            }
        }
        return maxVal;
    }


    public Matrix getGroundTruth(Model qFunction,
                                 double discountFactor)
    {
        Matrix yGT = Matrix.zeros(this.size(), 1);

        for (int i = 0; i < this.size(); ++i)
        {
            double R = this.getRewards().get(i, 0);
            double target = R;                         // default for terminal

            if (!this.getIsStateTerminalMask()[i])     // non‑terminal ⇒ add γ·max Q(s′,·)
            {
                try
                {
                    Matrix qNext = qFunction.forward(this.getNextStates().getRow(i));
                    target += discountFactor * max(qNext);
                }
                catch (Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
            yGT.set(i, 0, target);
        }
        return yGT;
    }


    public Pair<Matrix, Matrix> getTrainingData(Model qFunction,
                                                double discountFactor)
    {
        Matrix X = Matrix.zeros(this.size(), this.getPrevStates().getShape().getNumCols());
        try
        {
            for(int rIdx = 0; rIdx < this.size(); ++rIdx)
            {
                X.copySlice(rIdx, rIdx+1, 0, X.getShape().getNumCols(),
                            this.getPrevStates().getRow(rIdx));
            }
        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }
        Matrix YGt = this.getGroundTruth(qFunction, discountFactor);

        return new Pair<Matrix, Matrix>(X, YGt);
    }

}

