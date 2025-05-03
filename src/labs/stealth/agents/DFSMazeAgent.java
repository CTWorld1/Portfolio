package src.labs.stealth.agents;

import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;
import edu.cwru.sepia.environment.model.state.State.StateView;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

/**
 * DFS-based Maze Agent for pathfinding
 */
public class DFSMazeAgent extends MazeAgent {

    public DFSMazeAgent(int playerNum) {
        super(playerNum);
    }

    /**
     * Performs DFS to find a path from src to the goal.
     *
     * @param src   The starting vertex.
     * @param goal  The target vertex.
     * @param state The current state of the game.
     * @return A path ending at the goal, or null if unreachable.
     */
    @Override
    public Path search(Vertex src, Vertex goal, StateView state) {
        Stack<Path> stack = new Stack<>();
        Set<Vertex> visited = new HashSet<>();

        // Start DFS from the source vertex.
        stack.push(new Path(src, 1, null));
        visited.add(src);

        while (!stack.isEmpty()) {
            Path currentPath = stack.pop();
            Vertex currentVertex = currentPath.getDestination();

            // If we have reached the goal, return the path.
            if (currentVertex.equals(goal)) {
                return currentPath;
            }

            // Expand to neighboring vertices (depth-first).
            for (Vertex neighbor : currentVertex.getNeighbors()) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    stack.push(new Path(neighbor, 1, currentPath));
                }
            }
        }

        return null; // No path found.
    }
}
