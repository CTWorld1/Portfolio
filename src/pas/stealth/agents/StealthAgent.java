package src.pas.stealth.agents;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.util.Direction;

import java.util.*;

import edu.bu.pas.stealth.agents.AStarAgent;
import edu.bu.pas.stealth.agents.AStarAgent.AgentPhase;
import edu.bu.pas.stealth.agents.AStarAgent.ExtraParams;
import edu.bu.pas.stealth.graph.Path;
import edu.bu.pas.stealth.graph.Vertex;

public class StealthAgent extends AStarAgent {

    private static final double ENEMY_PENALTY = 300.0;
    private static final float RISK_PENALTY = 100.0f;

    private int footmanID = -1;
    private int enemyChebyshevSightLimit;
    private boolean townhallKilled;
    private Path plan;
    private AgentPhase phase;
    private Vertex startPosition;
    private Vertex simulatedPosition;
    private final Map<Integer, Vertex> lastArcherPositions;

    public StealthAgent(int playerNum) {
        super(playerNum);
        enemyChebyshevSightLimit = -1;
        townhallKilled = false;
        plan = null;
        phase = AgentPhase.INFILTRATE;
        startPosition = null;
        simulatedPosition = null;
        lastArcherPositions = new HashMap<>();
    }

    /* Logging helper method */
    private void log(String message) {
        System.out.println("[DEBUG] " + message);
    }

    public Vertex getStartPosition() {
        return startPosition;
    }

    public void setStartPosition(Vertex position) {
        this.startPosition = position;
    }

    public Path getPlan() {
        return plan;
    }

    public void setPlan(Path newPlan) {
        this.plan = newPlan;
    }

    public AgentPhase getPhase() {
        return phase;
    }

    public void setPhase(AgentPhase newPhase) {
        this.phase = newPhase;
    }

    @Override
    public Map<Integer, Action> initialStep(StateView state, HistoryView history) {
        log("Beginning initialStep...");
        super.initialStep(state, history);

        // Assign footman ID if not yet assigned.
        if (footmanID == -1) {
            for (Integer unitID : state.getUnitIds(getPlayerNumber())) {
                footmanID = unitID;
                break;
            }
        }
        log("Footman ID assigned: " + footmanID);

        // Set starting position.
        Vertex pos = getCurrentPosition(state);
        if (startPosition == null) {
            setStartPosition(pos);
            log("Start position set to: " + pos);
        }
        simulatedPosition = pos;
        log("Footman Starting Position: " + startPosition);

        // Find an enemy archer.
        UnitView firstArcher = null;
        for (Integer archerID : getEnemyUnitIDs(state)) {
            UnitView uv = state.getUnit(archerID);
            if (uv != null) {
                firstArcher = uv;
                log("Found first enemy archer with ID: " + archerID);
                break;
            }
        }
        if (firstArcher == null) {
            System.err.println("ERROR: Could not find any archers. Exiting.");
            System.exit(-1);
        }
        enemyChebyshevSightLimit = firstArcher.getTemplateView().getRange();
        log("Enemy Archer Sight Range: " + enemyChebyshevSightLimit);

        // Initialize enemy archer positions.
        for (Integer archerID : getEnemyUnitIDs(state)) {
            UnitView archer = state.getUnit(archerID);
            if (archer != null) {
                Vertex archerPos = new Vertex(archer.getXPosition(), archer.getYPosition());
                lastArcherPositions.put(archerID, archerPos);
            }
        }
        log("Current phase: " + phase);
        log("Current simulatedPosition: " + simulatedPosition);
        log("Current Path: " + plan);

        return null;
    }

    @Override
    public Map<Integer, Action> middleStep(StateView state, HistoryView history) {
        log("Beginning middleStep...");
        Map<Integer, Action> actions = new HashMap<>();

        // Sync simulatedPosition with actual position.
        simulatedPosition = getCurrentPosition(state);
        log("Updated simulatedPosition: " + simulatedPosition);

        // Attack if adjacent to townhall.
        Vertex thPos = getTownhallPosition(state);
        if (thPos != null && isAdjacent(simulatedPosition, thPos)) {
            Integer townhallID = getTownhallID(state);
            if (townhallID != null) {
                log("Attacking Townhall from " + simulatedPosition + " targeting townhall at: " + thPos);
                actions.put(getUnitID(), Action.createPrimitiveAttack(getUnitID(), townhallID));
                return actions; // Attack overrides movement.
            }
        }

        // Switch phase if townhall is destroyed.
        if (!townhallKilled && isTownhallDestroyed(state)) {
            townhallKilled = true;
            setPhase(AgentPhase.EXFILTRATE);
            setPlan(aStarSearch(simulatedPosition, getStartPosition(), state, null));
            log("Townhall destroyed! Switching to EXFILTRATE mode and recalculating path.");
        }

        // Recalculate path only if necessary.
        if (plan == null || shouldReplacePlan(state, null) || nextMoveIsDangerous(state)) {
            Vertex target;
            if (phase == AgentPhase.INFILTRATE) {
                target = getAttackPosition(state);
                if (target == null) {
                    target = getTownhallPosition(state); // Fallback target.
                }
            } else {
                target = getEscapePosition(state);
            }
            log("Recalculating A* path to target: " + target);
            setPlan(aStarSearch(simulatedPosition, target, state, null));
        }

        // Update enemy archer positions.
        for (Integer archerID : getEnemyUnitIDs(state)) {
            UnitView archer = state.getUnit(archerID);
            if (archer != null) {
                lastArcherPositions.put(archerID, new Vertex(archer.getXPosition(), archer.getYPosition()));
            }
        }

        // Determine next action.
        Action moveAction = getNextMoveAction(state);
        if (moveAction != null) {
            log("Issuing move action: " + moveAction);
            actions.put(getUnitID(), moveAction);
        } else {
            log("No valid move found! Clearing plan to force recalculation.");
            setPlan(null);
        }
        return actions;
    }

    @Override
    public Collection<Vertex> getNeighbors(Vertex v, StateView state, ExtraParams extraParams) {
        Collection<Vertex> neighbors = new ArrayList<>();
        int x = v.getXCoordinate(), y = v.getYCoordinate();
        int[] dx = {0, 0, 1, -1, 1, 1, -1, -1};
        int[] dy = {1, -1, 0, 0, 1, -1, 1, -1};

        for (int i = 0; i < dx.length; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (state.inBounds(nx, ny) && !state.isUnitAt(nx, ny) && !state.isResourceAt(nx, ny)) {
                neighbors.add(new Vertex(nx, ny));
            }
        }
        log(neighbors.size() + " neighbors found for " + v);
        return neighbors;
    }

    @Override
    public Path aStarSearch(Vertex src, Vertex dst, StateView state, ExtraParams extraParams) {
        if (src == null || dst == null) {
            log("A* search failed: Source or destination is null.");
            return null;
        }
        log("Starting A* search from " + src + " to " + dst);

        Map<Vertex, Double> gScore = new HashMap<>();
        Map<Vertex, Vertex> cameFrom = new HashMap<>();
        Set<Vertex> closedSet = new HashSet<>();
        PriorityQueue<Vertex> openSet = new PriorityQueue<>(Comparator.comparingDouble(
                v -> gScore.getOrDefault(v, Double.POSITIVE_INFINITY) + heuristic(v, dst, state)
        ));

        gScore.put(src, 0.0);
        openSet.add(src);

        while (!openSet.isEmpty()) {
            Vertex current = openSet.poll();
            if (closedSet.contains(current)) continue;
            closedSet.add(current);

            if (current.equals(dst)) {
                log("A* search completed. Path found.");
                return reconstructPath(cameFrom, current);
            }

            for (Vertex neighbor : getNeighbors(current, state, extraParams)) {
                if (closedSet.contains(neighbor)) continue;
                double tentativeG = gScore.get(current) + getEdgeWeight(current, neighbor, state, extraParams);
                if (tentativeG < gScore.getOrDefault(neighbor, Double.POSITIVE_INFINITY)) {
                    cameFrom.put(neighbor, current);
                    gScore.put(neighbor, tentativeG);
                    openSet.add(neighbor);
                }
            }
        }
        log("A* search failed: No path found.");
        return null;
    }

    private Path reconstructPath(Map<Vertex, Vertex> cameFrom, Vertex current) {
        Path path = new Path(current);
        while (cameFrom.containsKey(current)) {
            Vertex parent = cameFrom.get(current);
            path = new Path(parent, 1.0f, path);
            current = parent;
        }
        return path;
    }

    private double heuristic(Vertex a, Vertex b, StateView state) {
        double baseCost = Math.abs(a.getXCoordinate() - b.getXCoordinate()) +
                          Math.abs(a.getYCoordinate() - b.getYCoordinate());

        // Penalize paths near enemy units.
        for (Integer enemyID : getEnemyUnitIDs(state)) {
            UnitView enemy = state.getUnit(enemyID);
            if (enemy != null) {
                int enemyDistance = Math.max(Math.abs(enemy.getXPosition() - a.getXCoordinate()),
                                             Math.abs(enemy.getYPosition() - a.getYCoordinate()));
                if (enemyDistance <= enemyChebyshevSightLimit + 1) {
                    baseCost += ENEMY_PENALTY;
                }
            }
        }
        return baseCost;
    }

    @Override
    public float getEdgeWeight(Vertex src, Vertex dst, StateView state, ExtraParams extraParams) {
        float baseCost = 1.0f;
        float riskFactor = 0.0f;

        for (Integer enemyID : getEnemyUnitIDs(state)) {
            UnitView enemy = state.getUnit(enemyID);
            if (enemy != null) {
                Vertex enemyPos = new Vertex(enemy.getXPosition(), enemy.getYPosition());
                int chebyshev = Math.max(Math.abs(enemyPos.getXCoordinate() - dst.getXCoordinate()),
                                          Math.abs(enemyPos.getYCoordinate() - dst.getYCoordinate()));
                if (chebyshev <= enemyChebyshevSightLimit) {
                    riskFactor += RISK_PENALTY;
                }
            }
        }
        return baseCost + riskFactor;
    }

    @Override
    public boolean shouldReplacePlan(StateView state, ExtraParams extraParams) {
        if (plan == null || plan.getDestination() == null)
            return true;

        Path tempPlan = plan;
        while (tempPlan != null) {
            Vertex vertexToCheck = tempPlan.getDestination();

            if (state.isUnitAt(vertexToCheck.getXCoordinate(), vertexToCheck.getYCoordinate()) ||
                state.isResourceAt(vertexToCheck.getXCoordinate(), vertexToCheck.getYCoordinate())) {
                return true; // Blocked tile detected.
            }

            for (Integer enemyID : getEnemyUnitIDs(state)) {
                UnitView enemy = state.getUnit(enemyID);
                int chebyshev = Math.max(Math.abs(enemy.getXPosition() - vertexToCheck.getXCoordinate()),
                                          Math.abs(enemy.getYPosition() - vertexToCheck.getYCoordinate()));
                if (chebyshev <= enemyChebyshevSightLimit) {
                    return true; // Dangerous tile detected.
                }
            }
            tempPlan = tempPlan.getParentPath();
        }
        return false;
    }

    /**
     * Returns a valid attack position adjacent to the enemy townhall.
     */
    private Vertex getAttackPosition(StateView state) {
        Vertex townhall = getTownhallPosition(state);
        if (townhall == null) return null;
        Collection<Vertex> neighbors = getNeighbors(townhall, state, null);
        Vertex best = null;
        double bestHeuristic = Double.POSITIVE_INFINITY;
        for (Vertex v : neighbors) {
            double h = Math.abs(v.getXCoordinate() - simulatedPosition.getXCoordinate()) +
                       Math.abs(v.getYCoordinate() - simulatedPosition.getYCoordinate());
            if (h < bestHeuristic) {
                bestHeuristic = h;
                best = v;
            }
        }
        return best;
    }

    /**
     * Checks if two vertices are adjacent (Chebyshev distance of 1 or less).
     */
    private boolean isAdjacent(Vertex a, Vertex b) {
        int dx = Math.abs(a.getXCoordinate() - b.getXCoordinate());
        int dy = Math.abs(a.getYCoordinate() - b.getYCoordinate());
        return dx <= 1 && dy <= 1;
    }

    private boolean nextMoveIsDangerous(StateView state) {
        if (plan == null || plan.getDestination() == null)
            return false;
        Vertex nextStep = plan.getDestination();
        return isPositionDangerous(nextStep, state);
    }

    private boolean isPositionDangerous(Vertex position, StateView state) {
        for (Integer enemyID : getEnemyUnitIDs(state)) {
            UnitView enemy = state.getUnit(enemyID);
            if (enemy != null) {
                int chebyshev = Math.max(Math.abs(enemy.getXPosition() - position.getXCoordinate()),
                                          Math.abs(enemy.getYPosition() - position.getYCoordinate()));
                if (chebyshev <= enemyChebyshevSightLimit)
                    return true;
            }
        }
        return false;
    }

    private Direction getSafeDirection(StateView state) {
        Vertex currentPos = getCurrentPosition(state);
        Collection<Vertex> neighbors = getNeighbors(currentPos, state, null);
        Direction safestDirection = null;
        double lowestRisk = Double.POSITIVE_INFINITY;
        for (Vertex neighbor : neighbors) {
            double risk = 0.0;
            for (Integer enemyID : getEnemyUnitIDs(state)) {
                UnitView enemy = state.getUnit(enemyID);
                if (enemy != null) {
                    int chebyshev = Math.max(Math.abs(enemy.getXPosition() - neighbor.getXCoordinate()),
                                              Math.abs(enemy.getYPosition() - neighbor.getYCoordinate()));
                    if (chebyshev <= enemyChebyshevSightLimit)
                        risk += (enemyChebyshevSightLimit - chebyshev + 1) * 10000;
                }
            }
            if (risk < lowestRisk) {
                lowestRisk = risk;
                safestDirection = getDirectionToMoveTo(currentPos, neighbor);
            }
        }
        return safestDirection;
    }

    private boolean isTownhallDestroyed(StateView state) {
        return getTownhallID(state) == null;
    }

    private Integer getTownhallID(StateView state) {
        int enemy = 1 - getPlayerNumber();
        for (Integer unitID : state.getUnitIds(enemy)) {
            UnitView uv = state.getUnit(unitID);
            if (uv != null && "TownHall".equals(uv.getTemplateView().getName()))
                return unitID;
        }
        return null;
    }

    private Vertex getTownhallPosition(StateView state) {
        Integer thID = getTownhallID(state);
        if (thID != null) {
            UnitView uv = state.getUnit(thID);
            if (uv != null)
                return new Vertex(uv.getXPosition(), uv.getYPosition());
        }
        return null;
    }

    private Collection<Integer> getEnemyUnitIDs(StateView state) {
        int enemyPlayer = 1 - getPlayerNumber();
        List<Integer> enemies = new ArrayList<>();
        for (Integer unitID : state.getUnitIds(enemyPlayer)) {
            UnitView u = state.getUnit(unitID);
            if (u != null && !"TownHall".equals(u.getTemplateView().getName()))
                enemies.add(unitID);
        }
        return enemies;
    }

    private int getUnitID() {
        return footmanID;
    }

    private Vertex getEscapePosition(StateView state) {
        return getStartPosition();
    }

    /**
     * Returns the current position using the state.
     */
    private Vertex getCurrentPosition(StateView state) {
        UnitView footman = state.getUnit(getUnitID());
        if (footman == null) {
            System.err.println("ERROR: Footman unit not found in state!");
            return null;
        }
        Vertex actual = new Vertex(footman.getXPosition(), footman.getYPosition());
        if (!actual.equals(simulatedPosition)) {
            simulatedPosition = actual;
        }
        return simulatedPosition;
    }

    /**
     * Determines the next move action based on the current plan.
     */
    private Action getNextMoveAction(StateView state) {
        if (plan == null || plan.getDestination() == null) {
            log("No valid plan found. Waiting.");
            return null; // Wait if no plan exists.
        }
        Vertex currentPos = getCurrentPosition(state);
        // Get next step from the plan.
        Vertex nextStep = (plan.getParentPath() != null)
                        ? plan.getParentPath().getDestination()
                        : plan.getDestination();
        // If the next step is blocked, clear the plan.
        if (state.isUnitAt(nextStep.getXCoordinate(), nextStep.getYCoordinate()) ||
            state.isResourceAt(nextStep.getXCoordinate(), nextStep.getYCoordinate())) {
            log("[getNextMoveAction] Next step blocked! Recalculating path.");
            setPlan(null);
            return null;
        }
        Direction d = getDirectionToMoveTo(currentPos, nextStep);
        if (d != null) {
            simulatedPosition = nextStep;
            setPlan(plan.getParentPath());
            return Action.createPrimitiveMove(getUnitID(), d);
        }
        return null;
    }

    /**
     * Returns the direction from one vertex to another.
     */
    public Direction getDirectionToMoveTo(Vertex from, Vertex to) {
        int dx = to.getXCoordinate() - from.getXCoordinate();
        int dy = to.getYCoordinate() - from.getYCoordinate();
        log("Raw differences: dx = " + dx + ", dy = " + dy);
        if (dx == 0 && dy > 0)  return Direction.SOUTH;
        if (dx == 0 && dy < 0)  return Direction.NORTH;
        if (dx > 0 && dy == 0)  return Direction.EAST;
        if (dx < 0 && dy == 0)  return Direction.WEST;
        if (dx > 0 && dy > 0)   return Direction.SOUTHEAST;
        if (dx > 0 && dy < 0)   return Direction.NORTHEAST;
        if (dx < 0 && dy > 0)   return Direction.SOUTHWEST;
        if (dx < 0 && dy < 0)   return Direction.NORTHWEST;
        return null;
    }
}
