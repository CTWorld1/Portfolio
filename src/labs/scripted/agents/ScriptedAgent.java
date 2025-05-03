package src.labs.scripted.agents;

// SYSTEM IMPORTS
import edu.cwru.sepia.action.Action;                                        
import edu.cwru.sepia.agent.Agent;                                          
import edu.cwru.sepia.environment.model.history.History.HistoryView;        
import edu.cwru.sepia.environment.model.state.ResourceNode;                 
import edu.cwru.sepia.environment.model.state.ResourceNode.ResourceView;    
import edu.cwru.sepia.environment.model.state.ResourceType;                 
import edu.cwru.sepia.environment.model.state.State.StateView;              
import edu.cwru.sepia.environment.model.state.Unit.UnitView;                
import edu.cwru.sepia.util.Direction;                                       


import java.io.InputStream;
import java.io.OutputStream;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


public class ScriptedAgent extends Agent {

    private Integer myUnitId;            // ID of the unit we control
    private Integer enemyUnitId;         // ID of the enemy footman
    private Integer goldResourceNodeId;  // ID of the gold deposit

    public ScriptedAgent(int playerNum, String[] args) {
        super(playerNum);
        System.out.println("Constructed ScriptedAgent");
    }

    public final Integer getMyUnitId() { return this.myUnitId; }
    public final Integer getEnemyUnitId() { return this.enemyUnitId; }
    public final Integer getGoldResourceNodeId() { return this.goldResourceNodeId; }

    private void setMyUnitId(Integer i) { this.myUnitId = i; }
    private void setEnemyUnitId(Integer i) { this.enemyUnitId = i; }
    private void setGoldResourceNodeId(Integer i) { this.goldResourceNodeId = i; }

    @Override
    public Map<Integer, Action> initialStep(StateView state, HistoryView history) {
        // Discover friendly footman.
        Set<Integer> myUnitIds = new HashSet<>();
        for(Integer unitID : state.getUnitIds(this.getPlayerNumber())) {
            myUnitIds.add(unitID);
        }
        if(myUnitIds.size() != 1) {
            System.err.println("ERROR: Should control exactly 1 footman.");
            System.exit(-1);
        }

        // Discover enemy footman.
        Integer[] playerNumbers = state.getPlayerNumbers();
        if(playerNumbers.length != 2) {
            System.err.println("ERROR: Should only be 2 players in the game.");
            System.exit(-1);
        }
        Integer enemyPlayerNumber = (playerNumbers[0] == this.getPlayerNumber()) ? 
            playerNumbers[1] : playerNumbers[0];

        Set<Integer> enemyUnitIds = new HashSet<>();
        for(Integer unitID : state.getUnitIds(enemyPlayerNumber)) {
            enemyUnitIds.add(unitID);
        }
        if(enemyUnitIds.size() != 1) {
            System.err.println("ERROR: Enemy should have exactly 1 footman.");
            System.exit(-1);
        }

        // Set our fields
        this.setMyUnitId(myUnitIds.iterator().next());
        this.setEnemyUnitId(enemyUnitIds.iterator().next());

        // TODO: Find the gold resource node ID
        Integer goldMineId = null;
        for(ResourceView resource : state.getAllResourceNodes()) {
            // Check if it’s a gold mine (sometimes .getType() == ResourceNode.Type.GOLD_MINE)
            if(resource.getType() == ResourceNode.Type.GOLD_MINE) {
                goldMineId = resource.getID();
                break;
            }
        }
        this.setGoldResourceNodeId(goldMineId);
        System.out.println("Gold mine found with ID = " + goldMineId);

        // Now invoke middleStep for the first turn’s actions
        return this.middleStep(state, history);
    }

    @Override
    public Map<Integer, Action> middleStep(StateView state, HistoryView history) {
        Map<Integer, Action> actions = new HashMap<>();

        // If my unit is dead or gone, do nothing.
        UnitView myUnit = state.getUnit(myUnitId);
        if(myUnit == null) {
            return actions;
        }

        // If the enemy is dead, we do nothing (game should end soon).
        UnitView enemyUnit = state.getUnit(enemyUnitId);
        if(enemyUnit == null) {
            return actions;
        }

        // Get positions
        int myX = myUnit.getXPosition();
        int myY = myUnit.getYPosition();
        int enemyX = enemyUnit.getXPosition();
        int enemyY = enemyUnit.getYPosition();

        // Check if adjacent (Manhattan distance == 1 if horizontally or vertically adjacent)
        if(Math.abs(myX - enemyX) + Math.abs(myY - enemyY) == 1) {
            // Issue attack action
            actions.put(myUnitId, Action.createPrimitiveAttack(myUnitId, enemyUnitId));
        } else {
            // Otherwise, move step-by-step to get closer.
            // A simple approach: move in x-direction first, then y-direction.
            if(myX < enemyX) {
                actions.put(myUnitId, Action.createPrimitiveMove(myUnitId, Direction.EAST));
            } else if(myX > enemyX) {
                actions.put(myUnitId, Action.createPrimitiveMove(myUnitId, Direction.WEST));
            } else if(myY < enemyY) {
                actions.put(myUnitId, Action.createPrimitiveMove(myUnitId, Direction.SOUTH));
            } else if(myY > enemyY) {
                actions.put(myUnitId, Action.createPrimitiveMove(myUnitId, Direction.NORTH));
            }
        }

        return actions;
    }

    @Override
    public void terminalStep(StateView state, HistoryView history) {
        // Post-game analysis if desired.
        System.out.println("Game Over!");
    }

    @Override
    public void loadPlayerData(InputStream is) { }

    @Override
    public void savePlayerData(OutputStream os) { }
}
