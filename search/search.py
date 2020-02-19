# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal.
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    from util import Stack

    #creates frontier and root node using getStartState
    #lifo queue to keep track of next state(s) to explore
    frontier = Stack()
    frontier.push(problem.getStartState())

    #holds explored states
    explored = set()
    #holds final path to take
    finalPath = []
    #stack to hold next direction to take
    nextDirection = Stack()
    currState = frontier.pop()

    while not problem.isGoalState(currState):

        if currState not in explored:
            explored.add(currState)
            nextState = problem.getSuccessors(currState)

            for child, direction, cost in nextState:
                frontier.push(child)
                currPath = finalPath+[direction]
                nextDirection.push(currPath)

        currState = frontier.pop()
        finalPath = nextDirection.pop()

    return finalPath

    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue

    #creates frontier and root node using getStartState()
    #FIFO queue with node as the only element
    frontier = Queue()
    frontier.push(problem.getStartState())

    #holds explored states
    explored = set()
    #holds final path to take
    finalPath = []
    #queue to hold next direction to take
    nextDirection = Queue()
    currState = frontier.pop()

    while not problem.isGoalState(currState):

        if currState not in explored:
            explored.add(currState)
            nextState = problem.getSuccessors(currState)

            for child, direction, cost in nextState:
                frontier.push(child)
                currPath = finalPath+[direction]
                nextDirection.push(currPath)

        currState = frontier.pop()
        finalPath = nextDirection.pop()

    return finalPath

    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    #create a node for the state; assuming root has path cost of 0
    #creates frontier and root node using getStartState()
    #priority queue to help distinguish best next choice
    frontier = PriorityQueue()
    frontier.push(problem.getStartState(), 0)

    #holds explored states
    explored = set()
    #holds final path to take
    finalPath = []
    #queue to hold next direction to take
    nextDirection = PriorityQueue()
    currState = frontier.pop()

    while not problem.isGoalState(currState):
        if currState not in explored:
            explored.add(currState)
            nextState = problem.getSuccessors(currState)

            for child, direction, cost in nextState:
                currPath = finalPath+[direction]
                pathCost = problem.getCostOfActions(currPath)

                if child not in explored:
                    frontier.push(child, pathCost)
                    nextDirection.push(currPath, pathCost)

        currState = frontier.pop()
        finalPath = nextDirection.pop()

    return finalPath

    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    if problem.isGoalState(state):
        return -20
    else:
        return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import Queue, PriorityQueue

    #creates frontier and root node using getStartState()
    #priority queue to help distinguish best next choice
    frontier = PriorityQueue()
    frontier.push(problem.getStartState(), 0)

    #holds explored states
    explored = set()
    #holds final path to take
    finalPath = []
    #queue to hold next direction to take
    nextDirection = PriorityQueue()
    currState = frontier.pop()

    while not problem.isGoalState(currState):

        if currState not in explored:
            explored.add(currState)
            nextState = problem.getSuccessors(currState)

            for child, direction, cost in nextState:
                currPath = finalPath+[direction]
                pathCost = problem.getCostOfActions(currPath)+heuristic(child, problem)

                if child not in explored:
                    frontier.push(child, pathCost)
                    nextDirection.push(currPath, pathCost)

        currState = frontier.pop()
        finalPath = nextDirection.pop()

    return finalPath

    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
