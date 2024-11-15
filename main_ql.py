from utils.pddl import Domain, Problem
from utils.planning_world import PlanningWorld
from utils.blocksworld_game import BlocksworldGame
import pdb

if __name__ == "__main__":
    game = BlocksworldGame(num_blocks=5)
    #game.ql_learn_init()
    game.method_comparison()
