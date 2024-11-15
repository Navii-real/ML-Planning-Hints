# utils for playing a blocksworld game
from .pddl import Domain, Problem
from .planning_world import PlanningWorld
from .z3_planner import Z3Planner
from os.path import dirname, join
import numpy as np
import pdb
import random
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from time import time
import gc

class BlocksworldGame:

    def __init__(self, num_blocks=None):
        self.pddl_path = join(dirname(dirname(__file__)), "pddl", "blocksworld")
        self.domain = Domain.read_from_file(join(self.pddl_path, "domain.pddl"))
        self.total_ql_steps = 0
        self.total_classical_steps = 0
        if num_blocks:
            self.num_blocks = num_blocks
            self.random_instance(num_blocks)
        else:
            self.load_problem_file("medium.pddl")

    def load_problem_file(self, fname):
        self.problem = Problem.read_from_file(self.domain, join(self.pddl_path, fname))
        self.world = PlanningWorld(self.problem)

    # random n-block problem instance with goal to
    # stack all the blocks in ordered tower
    def random_instance(self, n):
        blocks = [chr(65+i) for i in range(n)]
        random.shuffle(blocks)
        top_blocks, init = [], []
        for block in blocks:
            place = random.choice(top_blocks + ["table"])
            if place == "table":
                init.append(("ontable", block))
            else:
                init.append(("on", block, place))
                top_blocks.remove(place)
            top_blocks.append(block)
        for block in top_blocks:
            init.append(("clear", block))
        init.append(("handempty",))
        blocks.sort()
        goal = [("on", blocks[i], blocks[i+1])
                for i in range(len(blocks) - 1)]
        self.problem = Problem("random", self.domain, blocks, tuple(init), tuple(goal))
        self.world = PlanningWorld(self.problem)

    def run_classical_planner(self):
        plan = self.run_iterative_planner()
        gc.collect()
        step = len(plan)
        self.run_known_steps_planner(step)
        gc.collect()
        hint_step = step//2
        # step numbers are 1-indexed
        hint = (plan[hint_step -1], hint_step)
        self.run_hint_planner(step, hint)

    def run_iterative_planner(self):
        self.display_state()
        print("Starting classical planner.")
        planner = Z3Planner(self.problem)
        plan = planner.iterative_plan()
        print("Optimal plan found with ", len(plan), " steps.")
        self.display_state()
        for step in plan:
            self.world.take_action(step[0], step[1:])
            print()
            self.display_state()
        return plan

    def run_known_steps_planner(self, steps):
        print("replanning with # steps known...")
        start = time()
        planner2 = Z3Planner(self.problem, steps)
        plan = planner2.plan()
        end = time()
        print(plan)
        print("total time: ", end - start)
        self.total_classical_steps = len(plan)

    def run_hint_planner(self, steps, hint):
        print("replanning with hint")
        start = time()
        planner3 = Z3Planner(self.problem, steps)
        planner3.smt_encode()
        plan = planner3.hint_plan([hint])
        end = time()
        print(plan)
        print("total time: ", end - start)

    def open_cli(self):
        print("Starting blocksworld game. 'quit' to quit")
        while True:
            self.display_state()
            print("heuristic:", self.world.heuristic(self.world.state_expr()))
            command = input("enter action --> ")
            if command == "quit":
                print("exiting game")
                break
            words = command.split()
            if len(command) == 0:
                continue
            if not words[0] in self.domain.actions:
                print("Unknown action " + words[0])
                continue
            action = words[0]
            args = words[1:]
            for arg in args:
                if not arg in self.problem.objects:
                    print("Unknown object " + arg)
                    continue
            if len(args) != self.domain.actions[action].arity:
                print("Wrong number of arguments for " + action)
                continue
            if self.world.is_legal(action, tuple(args)):
                self.world.take_action(action, tuple(args))
                if self.world.goal_satisfied():
                    self.display_state()
                    print("Goal Satisfied!")
                    print("Final state", self.world.state_expr())
                    break # TODO start new problem?
            else:
                print("Illegal action")

    def open_cli_num(self):
        print("Starting number input based cli")
        while True:
            self.display_state()
            for i in range(self.world.num_actions):
                print(i, self.world.action_expr(i))
            command = input("enter action")
            expr = self.world.action_expr(int(command))
            if self.world.is_legal(expr[0], expr[1:]):
                self.world.take_action(expr[0], expr[1:])
                if self.world.goal_satisfied():
                    self.display_state()
                    print("Goal Satisfied!")
                    print("Final state", self.world.state_expr())
                    break # TODO start new problem?
            else:
                print("Illegal action")

    # bad name, this function trains a q table from scratch
    # we will also have a funciton that read a q table from a file
    # perhaps we will call that ql_implement?
    def ql_learn_init(self):
        self.q_table = np.zeros((self.world.num_actions,1))
        self.mask = np.ones((self.world.num_actions,1))
        self.state_index = {}
        self.state_index[self.world.state_expr()] = 0
        print("Starting learning process")
        for i in range(self.world.num_actions):
            print(i, self.world.action_expr(i))

        e = 1 #epsilon
        a = .9 #alpha
        y = .9 #gamma
        t = 0
        epoch = 0
        while e > 0.03:
            #self.display_state()

            # choose action part
            state = self.state_index[self.world.state_expr()]
            if random.random()>e:
                command = np.argmax(np.transpose(self.q_table)[state])
                #print("Agent greedily chose", command)
            else:
                #print("Random action!")
                indices = np.where(np.transpose(self.mask)[state] == 1)[0]
                command = random.choice(indices)
                #print("Agent chose", command)
            # end of action selection

            expr = self.world.action_expr(int(command))
            if self.world.is_legal(expr[0], expr[1:]):
                # take action
                self.world.take_action(expr[0], expr[1:])

                # observe s'
                if not self.world.state_expr() in self.state_index:
                    #print("We're not in Kansas anymore!")
                    self.state_index[self.world.state_expr()] = len(self.state_index)
                    self.q_table = np.append(self.q_table, np.zeros((self.world.num_actions,1)), axis=1)
                    self.mask = np.append(self.mask, np.ones((self.world.num_actions,1)), axis=1)

                reward = 0
                if self.world.goal_satisfied():
                    self.display_state()
                    print("Goal Satisfied!")
                    #print("Final state", self.world.state_expr())
                    reward = 1
                # update q table
                max_a = np.amax(np.transpose(self.q_table)[self.state_index[self.world.state_expr()]] )
                self.q_table[command][state] += a * (reward + y* max_a - self.q_table[command][state])
                t += 1
                if self.world.goal_satisfied() or t>5000: #1000 for 6 blocks
                    t = 0
                    e *= .9997 #.9995 for 6 blocks
                    epoch += 1
                    print("Starting epoch ", epoch,"\nEpsilon is", e)
                    self.__init__(self.num_blocks)
                    if not self.world.state_expr() in self.state_index:
                        #print("We're not in Kansas anymore!")
                        self.state_index[self.world.state_expr()] = len(self.state_index)
                        self.q_table = np.append(self.q_table, np.zeros((self.world.num_actions,1)), axis=1)
                        self.mask = np.append(self.mask, np.ones((self.world.num_actions,1)), axis=1)
            else:
                #print("Illegal action")
                self.mask[int(command)][self.state_index[self.world.state_expr()]] = 0
        print(self.q_table)
        print(self.mask)
        np.save('q_table',self.q_table)
        np.save('q_mask',self.mask)
        pickle.dump(self.state_index, open("state_index", "wb"))
        # save state index too

    def test_q(self):
        self.q_table = np.load('q_table.npy')
        self.mask = np.load('q_mask.npy')
        self.state_index = pickle.load(open("state_index", "rb"))

        # load state index
        print(self.q_table)
        print(self.mask)
        steps = 0

        # Initialize the blocksworld
        self.__init__(self.num_blocks)
        if not self.world.state_expr() in self.state_index:
            print("We're not in Kansas anymore!")
            self.state_index[self.world.state_expr()] = len(self.state_index)
            self.q_table = np.append(self.q_table, np.zeros((self.world.num_actions, 1)), axis=1)
            self.mask = np.append(self.mask, np.ones((self.world.num_actions, 1)), axis=1)

        # Is goal satisfied?
        sat = False

        # Timing
        start = datetime.now()
        print(start)

        with open('test.txt', 'a') as f:
            print(start, file = f)

        greedy = True
        # While the goal isn't satisfied, choose actions and move through the problem
        while not sat:
            print("On step: ", steps)
            with open('test.txt', 'a') as f:
                print("On step: ", steps, file=f)
            self.display_state()

            # choose action part
            state = self.state_index[self.world.state_expr()]
            # Always greedy
            if(greedy):
                command = np.argmax(np.transpose(self.q_table)[state])
                print("Agent greedily chose", command)
            else:
                print("Randomly chose", command)

            with open('test.txt', 'a') as f:
                print("Agent greedily chose", command, file=f)
            # end of action selection

            # If the command is legal, perform the action
            expr = self.world.action_expr(int(command))
            if self.world.is_legal(expr[0], expr[1:]):
                # take action
                self.world.take_action(expr[0], expr[1:])
                if not self.world.state_expr() in self.state_index:
                    print("We're not in Kansas anymore!")
                    self.state_index[self.world.state_expr()] = len(self.state_index)
                    self.q_table = np.append(self.q_table, np.zeros((self.world.num_actions,1)), axis=1)
                    self.mask = np.append(self.mask, np.ones((self.world.num_actions,1)), axis=1)

                greedy = True
                # Is the goal now satisfied?
                if self.world.goal_satisfied():
                    self.display_state()
                    sat = True
                    print("Goal Satisfied!")
                    print("Final state", self.world.state_expr())
                    with open('test.txt', 'a') as f:
                        print("Goal Satisfied!", file=f)
                        print("Final state", self.world.state_expr(), file=f)
            else:
                print("Illegal action: something went wrong :(")
                self.mask[int(command)][self.state_index[self.world.state_expr()]] = 0
                indices = np.where(np.transpose(self.mask)[state] == 1)[0]
                command = random.choice(indices)
                greedy = False

                with open('test.txt', 'a') as f:
                    print("Illegal action: something went wrong :(", file=f)
            steps+=1
            if steps > 2000:
                print("Test dump!")
                steps = 100
                break
        end = datetime.now()
        print("total time taken: ", end-start)
        with open('test.txt', 'a') as f:
            print("total time taken: ", end - start, file=f)
            print("", file=f)
            print("", file=f)
        self.total_ql_steps = steps

    def method_comparison(self):
        # Comparing the total number of steps each method takes to solve the same problem
        plt.style.use('seaborn-whitegrid')
        ql_steps = np.zeros(50)
        y_ax = np.zeros(100)
        x_ax = np.zeros(100)
        i = 0
        for x in ql_steps:
            self.test_q()
            planner = Z3Planner(self.problem)
            plan = planner.iterative_plan()
            ql_steps[i] = int(self.total_ql_steps-len(plan))
            if(ql_steps[i] < 100):
                y_ax[int(ql_steps[i])]+=1
            x_ax[i] = i
            i+=1

        fig = plt.figure(figsize=(20,10))
        plt.bar(x_ax,y_ax)
        fig.suptitle('QLearning Deviation from Optimal Solution', fontsize=20)
        plt.xlabel("Difference in Steps from Optimal Solution")
        plt.ylabel("Frequency")
        plt.xlim(0,20)
        plt.xticks(range(0,21))
        plt.show()


    def display_state(self):
        state = self.world.state_expr()
        stacks = [(clause[1],) for clause in state if clause[0] == "ontable"]
        holding = [clause[1] for clause in state if clause[0] == "holding"]
        holding = "|"+holding[0]+"|" if len(holding) > 0 else ''
        rest = [clause for clause in state if clause[0] == "on"]
        while len(rest) > 0:
            prev_level = [stack[-1] for stack in stacks]
            next_clause = [clause for clause in rest if clause[2] in prev_level][0]
            for i in range(len(stacks)):
                if stacks[i][-1] == next_clause[2]:
                    stacks[i] = stacks[i] + (next_clause[1],)
            rest.remove(next_clause)
        height = max([len(stack) for stack in stacks])
        for y in range(height):
            idx = height - y - 1
            row = ["|"+stack[idx]+"|" if len(stack) > idx else "   "
                   for stack in stacks]
            endl = "" if idx == 0 else "\n"
            print("".join(row), end=endl)
        print("  (>"+holding+"<)===")
