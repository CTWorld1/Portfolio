package src.labs.cp;

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.Random;

// JAVA PROJECT IMPORTS
import edu.bu.cp.linalg.Matrix;
import edu.bu.cp.utils.Pair;

public class Dataset
{
    /*───────────────────────────────*/
    /*           BatchIterator       */
    /*───────────────────────────────*/
    public static class BatchIterator implements Iterator<Pair<Matrix, Matrix>>
    {
        private final Matrix X;
        private final Matrix YGt;
        private final long   batchSize;
        private final long   numBatches;
        private long         currentBatchIdx;

        public BatchIterator(final Matrix X,
                             final Matrix YGt,
                             final long   batchSize,
                             final long   numBatches)
        {
            this.X              = X;
            this.YGt           = YGt;
            this.batchSize      = batchSize;
            this.numBatches     = numBatches;
            this.currentBatchIdx = 0L;
        }

        /* getters */
        private Matrix getFullX()         { return this.X;        }
        private Matrix getFullYGt()       { return this.YGt;      }
        private long   getBatchSize()     { return this.batchSize;}
        private long   getNumBatches()    { return this.numBatches;}
        private long   getCurrentBatchIdx() { return this.currentBatchIdx; }

        /* setters */
        private void setCurrentBatchIdx(long idx) { this.currentBatchIdx = idx; }

        @Override
        public boolean hasNext() { return getCurrentBatchIdx() < getNumBatches(); }

        @Override
        public Pair<Matrix, Matrix> next()
        {
            int rIdxStart = (int)(getBatchSize() * getCurrentBatchIdx());

            /* -------- fixed upper‑index calculation -------- */
            int rIdxEnd   = Math.min(rIdxStart + (int)getBatchSize(),
                                     getFullX().getShape().getNumRows());
            /* ------------------------------------------------ */

            Matrix XBatch  = null;
            Matrix YGtBatch = null;

            try {
                XBatch  = getFullX() .getSlice(rIdxStart, rIdxEnd,
                                               0, getFullX().getShape().getNumCols());
                YGtBatch = getFullYGt().getSlice(rIdxStart, rIdxEnd,
                                                 0, getFullYGt().getShape().getNumCols());
            } catch (Exception e) {
                System.err.println("[ERROR] BatchIterator.next: caught");
                e.printStackTrace();
                System.exit(-1);
            }

            setCurrentBatchIdx(getCurrentBatchIdx() + 1);
            return new Pair<>(XBatch, YGtBatch);
        }
    }

    /*───────────────────────────────*/
    /*          Dataset body         */
    /*───────────────────────────────*/
    private final Matrix X;
    private final Matrix YGt;
    private final long   batchSize;
    private final Random rng;

    public Dataset(final Matrix X,
                   final Matrix YGt,
                   final long   batchSize,
                   final Random rng)
    {
        this.X         = X;
        this.YGt      = YGt;
        this.batchSize = batchSize;
        this.rng       = rng;
    }

    /* public helpers */
    public BatchIterator iterator()
    {
        return new BatchIterator(X, YGt, batchSize, size());
    }

    public void shuffle()
    {
        try {
            int randIdx;
            Matrix tmp;
            for (int idx = 0; idx < X.getShape().getNumRows(); ++idx) {
                randIdx = rng.nextInt(idx + 1);

                /* swap in X */
                tmp = X.getRow(randIdx);
                X.copySlice(randIdx, randIdx + 1, 0, X.getShape().getNumCols(),
                            X.getRow(idx));
                X.copySlice(idx, idx + 1, 0, X.getShape().getNumCols(), tmp);

                /* swap in Y */
                tmp = YGt.getRow(randIdx);
                YGt.copySlice(randIdx, randIdx + 1, 0, YGt.getShape().getNumCols(),
                              YGt.getRow(idx));
                YGt.copySlice(idx, idx + 1, 0, YGt.getShape().getNumCols(), tmp);
            }
        } catch (Exception e) {
            System.err.println("[ERROR] Dataset.shuffle: caught");
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public long size()
    {
        long numSamples = X.getShape().getNumRows();
        long numBatches = numSamples / batchSize;
        if (numSamples % batchSize != 0) numBatches += 1;
        return numBatches;
    }
}
