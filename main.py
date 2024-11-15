from utils.pddl import Domain, Problem
from utils.planning_world import PlanningWorld
from utils.blocksworld_game import BlocksworldGame
from utils.z3_planner import Z3Planner
import pdb

if __name__ == "__main__":
    game = BlocksworldGame(num_blocks=6)
    game.run_classical_planner()
