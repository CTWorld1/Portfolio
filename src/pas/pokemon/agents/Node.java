package pas.pokemon.agents;

import edu.bu.labs.pokemon.models.*;

import java.util.*;

public class Node {
    public enum NodeType { MAX, MIN, CHANCE }

    private BattleView state;
    private NodeType type;
    private List<Node> children;
    private double probability; // for CHANCE nodes
    private double utility; // heuristic value

    public Node(BattleView state, NodeType type) {
        this.state = state;
        this.type = type;
        this.children = new ArrayList<>();
        this.utility = evaluateState();
    }

    public void generateChildren() {
        if (this.type == NodeType.MAX) {
            // Generate possible moves for our team
            Team.TeamView myTeam = state.getTeam1View(); // Player
            Pokemon.PokemonView active = myTeam.getActivePokemonView();
            List<Move.MoveView> moves = active.getAvailableMoves();
    
            for (Move.MoveView move : moves) {
                if (Battle.canApplyMove(state, active, move, 0)) {
                    List<Pair<Double, Battle.BattleView>> outcomes = move.getPotentialEffects(state, 0, 1);
    
                    for (Pair<Double, Battle.BattleView> outcome : outcomes) {
                        Node child = new Node(outcome.getSecond(), NodeType.CHANCE);
                        child.probability = outcome.getFirst();
                        this.children.add(child);
                    }
                }
            }
    
        } else if (this.type == NodeType.MIN) {
            // Mirror the MAX logic for the opponent
            Team.TeamView oppTeam = state.getTeam2View();
            Pokemon.PokemonView active = oppTeam.getActivePokemonView();
            List<Move.MoveView> moves = active.getAvailableMoves();
    
            for (Move.MoveView move : moves) {
                if (Battle.canApplyMove(state, active, move, 1)) {
                    List<Pair<Double, Battle.BattleView>> outcomes = move.getPotentialEffects(state, 1, 0);
    
                    for (Pair<Double, Battle.BattleView> outcome : outcomes) {
                        Node child = new Node(outcome.getSecond(), NodeType.CHANCE);
                        child.probability = outcome.getFirst();
                        this.children.add(child);
                    }
                }
            }
    
        } else if (this.type == NodeType.CHANCE) {
            // Apply randomness, sleep/confusion/freeze logic, etc.
            // For now, just add the state itself as a deterministic child.
            this.children.add(new Node(this.state, NodeType.MAX));
        }
    }
    

    public double evaluateState() {
        double score = 0.0;
    
        Team.TeamView myTeam = state.getTeamView(0);      // Player
        Team.TeamView oppTeam = state.getTeamView(1);     // Opponent
    
        for (int i = 0; i < myTeam.size(); i++) {
            Pokemon.PokemonView p = myTeam.getPokemonView(i);
            if (p != null && !p.hasFainted()) {
                score += p.getCurrentStat(Stat.HP);
            }
        }
    
        for (int i = 0; i < oppTeam.size(); i++) {
            Pokemon.PokemonView p = oppTeam.getPokemonView(i);
            if (p != null && !p.hasFainted()) {
                score -= p.getCurrentStat(Stat.HP);
            }
        }
    
        return score;
    }
    

    public List<Node> getChildren() {
        return children;
    }

    public double getUtility() {
        return utility;
    }

    public NodeType getType() {
        return type;
    }

    public BattleView getState() {
        return state;
    }
}
