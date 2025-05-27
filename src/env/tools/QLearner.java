package tools;

import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt 
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private HashMap<Integer, double[][]> qTables; // a map for storing the qTables computed for different goals

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();
  }

/**
* Computes a Q matrix for the state space and action space of the lab, and against
* a goal description. For example, the goal description can be of the form [z1level, z2Level],
* where z1Level is the desired value of the light level in Zone 1 of the lab,
* and z2Level is the desired value of the light level in Zone 2 of the lab.
* For exercise 11, the possible goal descriptions are:
* [0,0], [0,1], [0,2], [0,3], 
* [1,0], [1,1], [1,2], [1,3], 
* [2,0], [2,1], [2,2], [2,3], 
* [3,0], [3,1], [3,2], [3,3].
*
*<p>
* HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
* to interact with the learning environment (here, the lab), e.g., to retrieve the
* applicable actions, perform an action at the lab during learning etc.
*</p>
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  episodesObj the number of episodes used for calculating the Q matrix
* @param  alphaObj the learning rate with range [0,1].
* @param  gammaObj the discount factor [0,1]
* @param epsilonObj the exploration probability [0,1]
* @param rewardObj the reward assigned when reaching the goal state
**/
  @OPERATION
  public void calculateQ(Object[] goalDescription , Object episodesObj, Object alphaObj, Object gammaObj, Object epsilonObj, Object rewardObj) {
    
    // ensure that the right datatypes are used
    Integer episodes = Integer.valueOf(episodesObj.toString());
    Double alpha = Double.valueOf(alphaObj.toString());
    Double gamma = Double.valueOf(gammaObj.toString());
    Double epsilon = Double.valueOf(epsilonObj.toString());
    Integer reward = Integer.valueOf(rewardObj.toString());

    int goalKey = Arrays.hashCode(goalDescription);
    double[][] qTable = qTables.getOrDefault(goalKey, initializeQTable());

    List<Object> goalDescList = Arrays.asList(goalDescription);
    Set<Integer> goalSet = new HashSet<>(lab.getCompatibleStates(goalDescList));

    Random rnd = new Random();
    int maxStepsPerEpisode = 100;
    

    for (int ep = 0; ep < episodes; ep++) {
      for (int i = 0; i < 3; i++) {
          lab.performAction(rnd.nextInt(actionCount));
      }
      int state = lab.readCurrentState();
      boolean done = goalSet.contains(state);
      int steps = 0;

      while (!done && steps < maxStepsPerEpisode) {
          steps++;

          List<Integer> actions = lab.getApplicableActions(state);
          if (actions.isEmpty()) break;
          int action;
          if (rnd.nextDouble() < epsilon) {
              action = actions.get(rnd.nextInt(actions.size()));
          } else {
              action = actions.get(0);
              double bestQ = qTable[state][action];
              for (int a : actions) {
                  if (qTable[state][a] > bestQ) {
                      bestQ = qTable[state][a];
                      action = a;
                  }
              }
          }

          tools.Action act = lab.getAction(action);
          lab.performAction(action);
          int nextState = lab.readCurrentState();
          boolean reached = goalSet.contains(nextState);

          double cost = 0;
          String tag = act.getActionTag();
          Object payload = act.getPayload()[0];
          if (tag.endsWith("#SetZ1Light") || tag.endsWith("#SetZ2Light")) {
              // turning a light on costs 50; turning it off costs 0
              if (Boolean.TRUE.equals(payload)) cost = 50;
          } else {
              // raising blinds costs 1; lowering costs 0
              if (Boolean.TRUE.equals(payload)) cost = 1;
          }

          double r = reached
                     ? (reward - cost)
                     : (-cost);

          double maxNextQ = 0;
          for (double qv : qTable[nextState]) {
              if (qv > maxNextQ) maxNextQ = qv;
          }
          qTable[state][action] += alpha * (r + gamma * maxNextQ - qTable[state][action]);

          state = nextState;
          done  = reached;
      }
  }

  // 5. Persist and report
  qTables.put(goalKey, qTable);
  LOGGER.info("Finished training Q table for goal " + Arrays.toString(goalDescription));

  printQTable(qTable);
  
  }

/**
* Returns information about the next best action based on a provided state and the QTable for
* a goal description. The returned information can be used by agents to invoke an action 
* using a ThingArtifact.
*
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  currentStateDescription the current state e.g. [2,2,true,false,true,true,2]
* @param  nextBestActionTag the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
* @param  nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
* @param nextBestActionPayload the (returned) payload of the next best action, e.g. true
**/

@OPERATION
/* 
public void getActionFromState(Object[] goalDescription, Object[] currentStateDescription,
    OpFeedbackParam<String> nextBestActionTag, OpFeedbackParam<Object[]> nextBestActionPayloadTags,
    OpFeedbackParam<Object[]> nextBestActionPayload) {
        
      // remove the following upon implementing Task 2.3!

      // sets the semantic annotation of the next best action to be returned 
      nextBestActionTag.set("http://example.org/was#SetZ1Light");

      // sets the semantic annotation of the payload of the next best action to be returned 
      Object payloadTags[] = { "Z1Light" };
      nextBestActionPayloadTags.set(payloadTags);

      // sets the payload of the next best action to be returned 
      Object payload[] = { true };
      nextBestActionPayload.set(payload);
    }
    */
    public void getActionFromState(Object[] goalDescription,
                                 Object[] currentStateDescription,
                                 OpFeedbackParam<String> nextBestActionTag,
                                 OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                 OpFeedbackParam<Object[]> nextBestActionPayload) {
    int goalKey = Arrays.hashCode(goalDescription);
    double[][] qTable = qTables.get(goalKey);
    if (qTable == null) {
      LOGGER.warning("No Q table for goal " + Arrays.toString(goalDescription));
      return;
    }

    List<Object> stateDescList = Arrays.asList(currentStateDescription);
    List<Integer> compatibles = lab.getCompatibleStates(stateDescList);
    if (compatibles.isEmpty()) {
      LOGGER.warning("No compatible state for description "
                     + Arrays.toString(currentStateDescription));
      return;
    }
    int state = compatibles.get(0);

    List<Integer> applicable = lab.getApplicableActions(state);
    if (applicable.isEmpty()) {
      LOGGER.warning("No applicable actions in state " + state);
      return;
    }
    int bestAction = applicable.get(0);
    double bestQ = qTable[state][bestAction];
    for (int a : applicable) {
      if (qTable[state][a] > bestQ) {
        bestQ = qTable[state][a];
        bestAction = a;
      }
    }

    Action act = lab.getAction(bestAction);
    nextBestActionTag.set(act.getActionTag());
    nextBestActionPayloadTags.set(act.getPayloadTags());
    nextBestActionPayload.set(act.getPayload());

    LOGGER.info("getActionFromState for: state=" + state + ", action=" + act + ", Q=" + bestQ);
  }
    

    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < qTable.length; i++) {
      System.out.print("From state " + i + ":  ");
     for (int j = 0; j < qTable[i].length; j++) {
      System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
  }

  /**
  * Initialize a Q matrix
  *
  * @return the Q matrix
  */
 private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++){
      for(int j = 0; j < actionCount; j++){
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }
}
