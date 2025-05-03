package src.labs.pokemon.agents;

import edu.bu.labs.pokemon.core.Agent;
import edu.bu.labs.pokemon.core.Battle.BattleView;
import edu.bu.labs.pokemon.core.Move.MoveView;
import edu.bu.labs.pokemon.traversal.Node;
import edu.bu.labs.pokemon.utils.Pair;
import edu.bu.labs.pokemon.core.Pokemon.PokemonView;


import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.*;

public class AlphaBetaAgent extends Agent {

    // Global debug flag; set to false to disable debugging output
    private static final boolean DEBUG = true;
    
    private static void debug(String message) {
        if (DEBUG) {
            System.out.println("DEBUG: " + message);
        }
    }
    
    /**
     * Custom move ordering.
     * For maximizing nodes we sort in descending order, and for minimizing nodes in ascending order.
     */

    private static class MoveOrderer {
        public static List<Node> order(List<Node> children) {
            if (children == null || children.isEmpty()) {
                return children;
            }

            int childDepth = children.get(0).getDepth();
            boolean isMaximizing = (childDepth % 2 == 0);

            Collections.sort(children, new Comparator<Node>() {
                @Override
                public int compare(Node a, Node b) {
                    double scoreA = evaluateMove(a);
                    double scoreB = evaluateMove(b);

                    return isMaximizing ? Double.compare(scoreB, scoreA) : Double.compare(scoreA, scoreB);
                }
            });

            return children;
        }

        private static double evaluateMove(Node node) {
            double score = node.getUtilityValue();
            MoveView move = node.getLastMoveView();

            if (move == null) return score;

            // **1. Prefer Higher-Damage Moves**
            if (move.getPower() != null) {
                score += move.getPower() * 2; // Extra weight for damage
            }

            // **2. Penalize Low-Accuracy Moves**
            if (move.getAccuracy() != null && move.getAccuracy() < 85) {
                score -= 10; // Avoid risky moves
            }

            // **3. Prioritize STAB (Same-Type Attack Bonus) Moves**
            PokemonView activePokemon = node.getCurrentPlayerTeamView().getPokemonView(0); // FIXED
            if (move.getType().equals(activePokemon.getType1()) || move.getType().equals(activePokemon.getType2())) {
                score += 15; // Boost for STAB
            }

            // **4. Favor Strong Special Moves for Pikachu (Electric) & Charmander (Fire)**
            if (move.getType().toString().equals("ELECTRIC") || move.getType().toString().equals("FIRE")) {
                score += 20; // Strong moves for Pikachu/Charmander
            }

            // **5. Prioritize Moves with Higher Priority**
            score += move.getPriority() * 3;

            return score;
        }
    }




    
    private class AlphaBetaSearcher implements Callable<Pair<MoveView, Long>> {

        private final Node rootNode;
        private final int maxDepth;
        private final int myTeamIdx;

        public AlphaBetaSearcher(Node rootNode, int maxDepth, int myTeamIdx) {
            this.rootNode = rootNode;
            this.maxDepth = maxDepth;
            this.myTeamIdx = myTeamIdx;
        }
        
        /**
         * Modified alpha-beta pruning: uses the current player's team index (like in minimax)
         * to decide if the node is maximizing or minimizing, and always returns the immediate child
         * (i.e. the move) that produced the best outcome.
         */
        public Node normalAlphaBeta(Node node, double alpha, double beta) {
            // Base case: terminal node or depth limit reached.
            if (node.isTerminal() || node.getDepth() >= this.maxDepth) {
                return node;
            }
            
            // Decide MAX/MIN based on the current player's team index.
            boolean isMaximizing = (node.getCurrentPlayerTeamIdx() == this.myTeamIdx);
            double bestValue = isMaximizing ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
            Node bestChild = null;
            
            // Order children to help with pruning.
            List<Node> children = MoveOrderer.order(node.getChildren());
            if (children.isEmpty()) {
                return node;
            }
            
            for (Node child : children) {
                // Recursively evaluate the child.
                Node candidate = normalAlphaBeta(child, alpha, beta);
                double candidateValue = candidate.getUtilityValue();
                
                if (isMaximizing) {
                    if (candidateValue > bestValue) {
                        bestValue = candidateValue;
                        // Always select the immediate child that leads to the best outcome.
                        bestChild = child;
                    }
                    alpha = Math.max(alpha, bestValue);
                } else {
                    if (candidateValue < bestValue) {
                        bestValue = candidateValue;
                        bestChild = child;
                    }
                    beta = Math.min(beta, bestValue);
                }
                
                if (alpha >= beta) {
                    break; // Prune remaining branches.
                }
            }
            
            // Propagate the best utility value upward.
            node.setUtilityValue(bestValue);
            return bestChild;
        }

        public Node minimaxFirstLayerWhenImNotRoot(Node node, double alpha, double beta) {
            Node bestGrandChild = null;
            double bestUtilityValue = Double.POSITIVE_INFINITY;
            for (Node child : node.getChildren()) {
                if (!child.isTerminal()) {
                    Node grandChild = this.normalAlphaBeta(child, alpha, beta);
                    double util = grandChild.getUtilityValue();
                    if (util < bestUtilityValue) { // Opponent minimizes utility.
                        bestUtilityValue = util;
                        bestGrandChild = grandChild;
                    }
                }
            }
            return bestGrandChild;
        }

        public Node alphaBetaSearch(Node node, double alpha, double beta) {
            return this.myTeamIdx == 0 ?
                   this.normalAlphaBeta(node, alpha, beta) :
                   this.minimaxFirstLayerWhenImNotRoot(node, alpha, beta);
        }

        @Override
        public Pair<MoveView, Long> call() throws Exception {
            debug("Search started.");
            double startTime = System.nanoTime();

            Node resultNode = this.alphaBetaSearch(this.rootNode, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
            MoveView move = (resultNode != null) ? resultNode.getLastMoveView() : null;

            double endTime = System.nanoTime();
            long durationInMs = (long)((endTime - startTime) / 1000000);
            debug("Move selected = " + move + " in " + durationInMs + " ms");
            return new Pair<>(move, durationInMs);
        }
    }

    private final int maxDepth;
    private long maxThinkingTimePerMoveInMS;

    public AlphaBetaAgent() {
        super();
        this.maxThinkingTimePerMoveInMS = 180000; // 3 minutes
        this.maxDepth = 1000;
        debug("Agent constructed with maxDepth = " + this.maxDepth);
    }

    public int getMaxDepth() { return this.maxDepth; }
    public long getMaxThinkingTimePerMoveInMS() { return this.maxThinkingTimePerMoveInMS; }

    @Override
    public Integer chooseNextPokemon(BattleView view) {
        for (int idx = 0; idx < this.getMyTeamView(view).size(); ++idx) {
            if (!this.getMyTeamView(view).getPokemonView(idx).hasFainted()) {
                return idx;
            }
        }
        return null;
    }

    @Override
    public MoveView getMove(BattleView battleView) {
        debug("Starting move selection.");

        ExecutorService backgroundThreadManager = Executors.newSingleThreadExecutor();
        MoveView move = null;
        Node rootNode = new Node(battleView, this.getMyTeamIdx(), 0, 0);

        // **Dynamically adjust depth based on time and move complexity**
        int availableMoves = rootNode.getChildren().size();
        int dynamicDepth = calculateDynamicDepth(availableMoves, this.getMaxThinkingTimePerMoveInMS());

        AlphaBetaSearcher searcher = new AlphaBetaSearcher(rootNode, dynamicDepth, this.getMyTeamIdx());
        Future<Pair<MoveView, Long>> future = backgroundThreadManager.submit(searcher);

        try {
            Pair<MoveView, Long> moveAndDuration = future.get(this.getMaxThinkingTimePerMoveInMS(), TimeUnit.MILLISECONDS);
            move = moveAndDuration.getFirst();
        } catch (TimeoutException e) {
            System.err.println("Timeout!");
            debug("Timeout reached. Team [" + (this.getMyTeamIdx() + 1) + "] loses!");
            System.exit(-1);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            System.exit(-1);
        } finally {
            backgroundThreadManager.shutdownNow();
            debug("Background thread shutdown.");
        }
        return move;
    }

    /**
     * Adjusts search depth dynamically based on the number of available moves and time constraints.
     */
    private int calculateDynamicDepth(int moveCount, long maxTimeMS) {
        if (moveCount <= 2) {
            return Math.min(20, this.maxDepth); // Small decision space, go deeper
        } else if (moveCount <= 5) {
            return Math.min(12, this.maxDepth); // Medium complexity, medium depth
        } else if (moveCount <= 10) {
            return Math.min(8, this.maxDepth);  // Larger space, shallower depth
        } else {
            return Math.min(5, this.maxDepth);  // Many moves, keep it shallow
        }
    }

}
