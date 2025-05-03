package src.labs.stealth.agents;

import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;
import edu.cwru.sepia.environment.model.state.State.StateView;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

/**
 * BFS-based Maze Agent for pathfinding
 */
public class BFSMazeAgent extends MazeAgent {

    public BFSMazeAgent(int playerNum) {
        super(playerNum);
    }

    /**
     * Performs BFS to find the shortest path from src to the goal.
     *
     * @param src   The starting vertex.
     * @param goal  The target vertex.
     * @param state The current state of the game.
     * @return The shortest path ending at the goal, or null if unreachable.
     */
    @Override
    public Path search(Vertex src, Vertex goal, StateView state) {
        Queue<Path> queue = new LinkedList<>();
        Set<Vertex> visited = new HashSet<>();

        // Start BFS from the source vertex.
        queue.add(new Path(src, 1, null));
        visited.add(src);

        while (!queue.isEmpty()) {
            Path currentPath = queue.poll();
            Vertex currentVertex = currentPath.getDestination();

            // If we have reached the goal, return the path.
            if (currentVertex.equals(goal)) {
                return currentPath;
            }

            // Expand to neighboring vertices.
            for (Vertex neighbor : currentVertex.getNeighbors()) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.add(new Path(neighbor, 1, currentPath));
                }
            }
        }

        return null; // No path found.
    }
}
