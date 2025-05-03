package src.labs.stealth.agents;

import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.util.Direction;

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Dijkstra's Algorithm-based Maze Agent
 */
public class DijkstraMazeAgent extends MazeAgent {

    public DijkstraMazeAgent(int playerNum) {
        super(playerNum);
    }

    /**
     * Performs Dijkstra's Algorithm to find the shortest path from src to the goal.
     *
     * @param src   The starting vertex.
     * @param goal  The target vertex.
     * @param state The current state of the game.
     * @return The shortest path ending at the goal, or null if unreachable.
     */
    @Override
    public Path search(Vertex src, Vertex goal, StateView state) {
        PriorityQueue<Path> queue = new PriorityQueue<>(Comparator.comparingDouble(Path::getCost));
        Map<Vertex, Double> minCost = new HashMap<>(); // Tracks minimum cost to reach each vertex.
        Set<Vertex> visited = new HashSet<>(); // Avoid reprocessing nodes.

        // Start with the source vertex.
        queue.add(new Path(src, 0, null));
        minCost.put(src, 0.0);

        while (!queue.isEmpty()) {
            Path currentPath = queue.poll();
            Vertex currentVertex = currentPath.getDestination();

            // If we have reached the goal, return the path.
            if (currentVertex.equals(goal)) {
                return currentPath;
            }

            if (visited.contains(currentVertex)) {
                continue; // Skip already processed nodes.
            }
            visited.add(currentVertex);

            // Expand to neighboring vertices.
            for (Vertex neighbor : currentVertex.getNeighbors()) {
                if (visited.contains(neighbor)) {
                    continue;
                }

                // Compute movement cost based on direction.
                Direction moveDirection = getDirectionToMoveTo(currentVertex, neighbor);
                double moveCost = getMoveCost(moveDirection);

                double newCost = minCost.getOrDefault(currentVertex, Double.MAX_VALUE) + moveCost;

                if (newCost < minCost.getOrDefault(neighbor, Double.MAX_VALUE)) {
                    minCost.put(neighbor, newCost);
                    queue.add(new Path(neighbor, newCost, currentPath)); // Extend the path.
                }
            }
        }

        return null; // No path found.
    }

    /**
     * Helper function to determine movement cost based on direction.
     *
     * @param dir The movement direction.
     * @return The corresponding movement cost.
     */
    private double getMoveCost(Direction dir) {
        switch (dir) {
            case EAST:
            case WEST:
                return 5.0;
            case NORTH:
                return 5.0;
            case SOUTH:
                return 1.0;
            case UP:
                return 10.0;
            case NORTHWEST:
                return Math.sqrt(Math.pow(getMoveCost(Direction.NORTH), 2) + Math.pow(getMoveCost(Direction.WEST), 2));
            case NORTHEAST:
                return Math.sqrt(Math.pow(getMoveCost(Direction.NORTH), 2) + Math.pow(getMoveCost(Direction.EAST), 2));
            case SOUTHWEST:
                return Math.sqrt(Math.pow(getMoveCost(Direction.SOUTH), 2) + Math.pow(getMoveCost(Direction.WEST), 2));
            case SOUTHEAST:
                return Math.sqrt(Math.pow(getMoveCost(Direction.SOUTH), 2) + Math.pow(getMoveCost(Direction.EAST), 2));
            default:
                return Double.MAX_VALUE; // Invalid move.
        }
    }

    /**
     * Helper function to check if a vertex is a direct neighbor of the goal.
     *
     * @param v    The current vertex.
     * @param goal The goal vertex.
     * @return True if v is adjacent to goal, false otherwise.
     */
    private boolean isNeighbor(Vertex v, Vertex goal) {
        return goal.getNeighbors().contains(v);
    }
}
