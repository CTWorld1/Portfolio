package src.labs.rl.maze.agents;

// SYSTEM IMPORTS
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.util.Direction;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

// JAVA PROJECT IMPORTS
import edu.bu.labs.rl.maze.agents.StochasticAgent;
import edu.bu.labs.rl.maze.agents.StochasticAgent.RewardFunction;
import edu.bu.labs.rl.maze.agents.StochasticAgent.TransitionModel;
import edu.bu.labs.rl.maze.utilities.Coordinate;
import edu.bu.labs.rl.maze.utilities.Pair;

public class ValueIterationAgent extends StochasticAgent
{
    /* --------------------------------------------------------------------
       Tunable parameters
    -------------------------------------------------------------------- */
    public  static final double GAMMA   = 0.1;     // discount factor
    public  static final double EPSILON = 1e-6;    // convergence tolerance

    /* --------------------------------------------------------------------
       Internal state
    -------------------------------------------------------------------- */
    private Map<Coordinate, Double> utilities;

    public ValueIterationAgent(int playerNum)
    {
        super(playerNum);
        this.utilities = null;
    }

    /* Getters / setters -------------------------------------------------- */
    public  Map<Coordinate, Double> getUtilities()          { return utilities; }
    private void                    setUtilities(Map<Coordinate, Double> u)
    { this.utilities = u; }

    /* --------------------------------------------------------------------
       Helper utilities
    -------------------------------------------------------------------- */
    public boolean isTerminalState(Coordinate c)
    {
        return c.equals(StochasticAgent.POSITIVE_TERMINAL_STATE)
            || c.equals(StochasticAgent.NEGATIVE_TERMINAL_STATE);
    }

    /** Returns a map of every free (non-resource) square to 0.0. */
    public Map<Coordinate, Double> getZeroMap(StateView state)
    {
        Map<Coordinate, Double> m = new HashMap<>();
        for (int x = 0; x < state.getXExtent(); ++x)
            for (int y = 0; y < state.getYExtent(); ++y)
                if (!state.isResourceAt(x, y))
                    m.put(new Coordinate(x, y), 0.0);
        return m;
    }

    /* ********************************************************************
       VALUE-ITERATION  (Task-1 of the lab)
    ******************************************************************** */
    public void valueIteration(StateView state)
    {
        /* 0. Initialisation ------------------------------------------------*/
        Map<Coordinate, Double> Uprev = getZeroMap(state);   // U₀
        Map<Coordinate, Double> Unew  = new HashMap<>(Uprev.size());

        final double threshold = EPSILON * (1.0 - GAMMA) / GAMMA;
        double delta;                       // maximum update in a sweep

        /* 1. Repeat Bellman updates until change ≤ threshold --------------*/
        do {
            delta = 0.0;

            for (Coordinate s : Uprev.keySet())
            {
                /* Terminal states keep their reward ----------------------*/
                if (isTerminalState(s))
                {
                    double u = RewardFunction.getReward(s);
                    Unew.put(s, u);
                    delta = Math.max(delta, Math.abs(u - Uprev.get(s)));
                    continue;
                }

                /* Non-terminal: compute max expected utility among actions */
                double maxActionUtility = Double.NEGATIVE_INFINITY;

                for (Direction a : TransitionModel.CARDINAL_DIRECTIONS)
                {
                    double expectedU = 0.0;
                    for (Pair<Coordinate, Double> t :
                         TransitionModel.getTransitionProbs(state, s, a))
                    {
                        expectedU += t.getSecond() * Uprev.get(t.getFirst());
                    }
                    if (expectedU > maxActionUtility)
                        maxActionUtility = expectedU;
                }

                /* Bellman update ----------------------------------------*/
                double updatedU =
                        RewardFunction.getReward(s) + GAMMA * maxActionUtility;

                Unew.put(s, updatedU);

                /* Track largest change ----------------------------------*/
                delta = Math.max(delta, Math.abs(updatedU - Uprev.get(s)));
            }

            /* 2. Prepare for next sweep ----------------------------------*/
            Uprev = new HashMap<>(Unew);
            Unew.clear();

        } while (delta > threshold);

        /* 3. Save converged utilities ------------------------------------*/
        setUtilities(Uprev);
    }

    /* ********************************************************************
       After utilities converge, derive a greedy policy.
    ******************************************************************** */
    @Override
    public void computePolicy(StateView state, HistoryView history)
    {
        this.valueIteration(state);

        Map<Coordinate, Direction> policy = new HashMap<>();

        for (Coordinate c : this.getUtilities().keySet())
        {
            double    bestActionUtility = Double.NEGATIVE_INFINITY;
            Direction bestDirection     = null;

            for (Direction d : TransitionModel.CARDINAL_DIRECTIONS)
            {
                double actionUtility = 0.0;
                for (Pair<Coordinate, Double> t :
                     TransitionModel.getTransitionProbs(state, c, d))
                {
                    actionUtility += t.getSecond() * this.getUtilities().get(t.getFirst());
                }
                if (actionUtility > bestActionUtility)
                {
                    bestActionUtility = actionUtility;
                    bestDirection     = d;
                }
            }
            policy.put(c, bestDirection);
        }

        this.setPolicy(policy);
    }
}
