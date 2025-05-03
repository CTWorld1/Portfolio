package src.pas.pokemon.agents;

import edu.bu.pas.pokemon.core.Agent;
import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Team;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.utils.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Example TreeTraversalAgent that performs a single-ply expectimax search.
 * Adjust or expand as needed for deeper searches, confusion checks, etc.
 */
public class TreeTraversalAgent extends Agent {

    // -------------------------------------------------------
    // Nested class: StochasticTreeSearcher
    // -------------------------------------------------------
    private class StochasticTreeSearcher implements Callable<Pair<MoveView,Long>> {

        private final BattleView rootView;
        private final int maxDepth;
        private final int myTeamIdx;

        public StochasticTreeSearcher(BattleView rootView, int maxDepth, int myTeamIdx) {
            this.rootView = rootView;
            this.maxDepth = maxDepth;
            this.myTeamIdx = myTeamIdx;
        }

        @Override
        public Pair<MoveView, Long> call() throws Exception {
            double startTime = System.nanoTime();

            MoveView chosenMove = stochasticTreeSearch(rootView);

            double endTime = System.nanoTime();
            long durationMs = (long)((endTime - startTime) / 1_000_000.0);

            return new Pair<>(chosenMove, durationMs);
        }

        /**
         * Performs a single-ply "expectimax"-style search. For each legal move:
         *   1) get all possible (prob, nextState) from getPotentialEffects(...)
         *   2) compute expected utility
         *   3) pick the best move
         */
        private MoveView stochasticTreeSearch(BattleView view) {
            // 1) Get all legal moves for our active Pokemon
            List<MoveView> legalMoves = getLegalMovesForTeam(view, myTeamIdx);
            if (legalMoves.isEmpty()) {
                return null;
            }

            MoveView bestMove = null;
            double bestEV = Double.NEGATIVE_INFINITY;

            // 2) Evaluate each move's outcomes
            for (MoveView mv : legalMoves) {
                double ev = getExpectedValueOfMove(view, mv, myTeamIdx);
                if (ev > bestEV) {
                    bestEV = ev;
                    bestMove = mv;
                }
            }

            return bestMove;
        }

        /**
         * Sum(prob * utility(nextState)) over all nextStates from getPotentialEffects.
         */
        private double getExpectedValueOfMove(BattleView current, MoveView move, int userIdx) {
            int oppIdx = 1 - userIdx; // 2-team assumption

            // NOTE: your library requires the 3-arg version
            List<Pair<Double, BattleView>> outcomes = move.getPotentialEffects(current, userIdx, oppIdx);

            double total = 0;
            for (Pair<Double, BattleView> pair : outcomes) {
                double prob = pair.getFirst();
                BattleView nextState = pair.getSecond();

                double util = evaluateState(nextState, userIdx);
                total += prob * util;
            }
            return total;
        }

        /**
         * Very simple "my total HP minus opponent's total HP" heuristic.
         */
        private double evaluateState(BattleView state, int myIdx) {
            double myHP   = getTeamHP(state, myIdx);
            double oppHP  = getTeamHP(state, 1 - myIdx);
            return myHP - oppHP;
        }

        private double getTeamHP(BattleView state, int teamIdx) {
            TeamView tv = state.getTeamView(teamIdx);
            double sum = 0;
            for (int i = 0; i < tv.size(); i++) {
                PokemonView poke = tv.getPokemonView(i);
                if (poke != null && !poke.hasFainted()) {
                    // Instead of getCurrentHP(), we do:
                    sum += poke.getCurrentStat(Stat.HP);
                }
            }
            return sum;
        }

        /**
         * Gathers all "available" moves for the current active Pokemon, then
         * filters by canApplyMove(...) so we only keep truly legal moves.
         */
        private List<MoveView> getLegalMovesForTeam(BattleView view, int teamIdx) {
            TeamView myTeamView = view.getTeamView(teamIdx);
            if (myTeamView == null || myTeamView.getActivePokemonIdx() < 0)
                return new ArrayList<>();
            PokemonView activePoke = myTeamView.getActivePokemonView();
            if (activePoke == null || activePoke.hasFainted())
                return new ArrayList<>();

            // getAvailableMoves(): includes only moves with PP>0 or Struggle
            List<MoveView> moves = activePoke.getAvailableMoves();
            List<MoveView> legal = new ArrayList<>();

            for (MoveView mv : moves) {
                // Use your library's static check in Battle:
                if (Battle.canApplyMove(view, activePoke, mv, teamIdx)) {
                    legal.add(mv);
                }
            }
            return legal;
        }
    }

    // -------------------------------------------------------
    // Fields
    // -------------------------------------------------------
    private final int maxDepth;
    private final long maxThinkingTimePerMoveInMS;

    public TreeTraversalAgent() {
        super();
        // Example: 6 min time limit
        this.maxThinkingTimePerMoveInMS = 180000L * 2; 
        // Example: use large depth if you plan to prune heavily or do 1-ply
        this.maxDepth = 50; 
    }

    public int  getMaxDepth()                   { return this.maxDepth; }
    public long getMaxThinkingTimePerMoveInMS() { return this.maxThinkingTimePerMoveInMS; }

    // -------------------------------------------------------
    // If forced to choose next Pokemon after fainting
    // -------------------------------------------------------
    @Override
    public Integer chooseNextPokemon(BattleView view) {
        // Naive approach: pick first non-fainted
        TeamView myTeamView = getMyTeamView(view);
        if (myTeamView != null) {
            for (int i = 0; i < myTeamView.size(); i++) {
                PokemonView poke = myTeamView.getPokemonView(i);
                if (poke != null && !poke.hasFainted()) {
                    return i;
                }
            }
        }
        return null; // no living Pokemon left
    }

    // -------------------------------------------------------
    // Main method the engine calls to get our move
    // -------------------------------------------------------
    @Override
    public MoveView getMove(BattleView battleView) {

        ExecutorService executor = Executors.newSingleThreadExecutor();
        StochasticTreeSearcher searcher =
            new StochasticTreeSearcher(battleView, maxDepth, getMyTeamIdx());

        MoveView chosenMove = null;
        try {
            // time-limited call
            Future<Pair<MoveView,Long>> future = executor.submit(searcher);
            Pair<MoveView,Long> result = future.get(
                this.getMaxThinkingTimePerMoveInMS(),
                TimeUnit.MILLISECONDS
            );

            chosenMove = result.getFirst();
            long durationMs = result.getSecond();
            System.out.println("TreeTraversalAgent (Team " + (getMyTeamIdx() + 1) +
                               ") decided in " + durationMs + " ms");
        }
        catch (TimeoutException te) {
            System.err.println("TIMEOUT: TreeTraversalAgent for team " + (getMyTeamIdx()+1) + " ran out of time!");
            System.exit(-1);
        }
        catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        finally {
            executor.shutdownNow();
        }

        // If the search returns null, pick some fallback (like first legal move)
        if (chosenMove == null) {
            chosenMove = getFallbackMove(battleView);
        }

        return chosenMove;
    }

    /**
     * Simple fallback: picks first legal move if the search returns null.
     */
    private MoveView getFallbackMove(BattleView view) {
        List<MoveView> moves = getLegalMoves(view, getMyTeamIdx());
        if (!moves.isEmpty()) {
            return moves.get(0);
        }
    
        // Fallback #2: Try to get *any* move from available moves (even if not legal)
        TeamView tv = view.getTeamView(getMyTeamIdx());
        if (tv != null && tv.getActivePokemonIdx() >= 0) {
            PokemonView poke = tv.getActivePokemonView();
            if (poke != null && !poke.hasFainted()) {
                List<MoveView> allMoves = poke.getAvailableMoves();
                if (!allMoves.isEmpty()) {
                    return allMoves.get(0); // Pick the first available move
                }
            }
        }
    
        // As a last resort, return null (battle engine may treat this as "do nothing")
        return null;
    }
    
    /**
     * Helper that returns all legal moves for your active Pokemon,
     * for a fallback scenario.
     */
    private List<MoveView> getLegalMoves(BattleView view, int teamIdx) {
        TeamView tv = view.getTeamView(teamIdx);
        if (tv == null || tv.getActivePokemonIdx() < 0) return new ArrayList<>();

        PokemonView poke = tv.getActivePokemonView();
        if (poke == null || poke.hasFainted()) return new ArrayList<>();

        // getAvailableMoves() might already skip 0-PP moves
        List<MoveView> moves = poke.getAvailableMoves();
        List<MoveView> legal = new ArrayList<>();
        for (MoveView mv : moves) {
            if (Battle.canApplyMove(view, poke, mv, teamIdx)) {
                legal.add(mv);
            }
        }
        return legal;
    }
}
