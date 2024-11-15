# Generates performance data for the classical planner
# and hint-based planner on random problem instances
# and plots the results

from utils.blocksworld_game import BlocksworldGame
from utils.z3_planner import Z3Planner
import gc
from time import time
import pdb
import matplotlib.pyplot as plt
import numpy as np

def main():
    classical_times = []
    good_hint_times = []
    bad_hint_times = []
    start_n, end_n = 4, 6
    num_trials = 20
    game = BlocksworldGame(num_blocks=start_n)

    for n in range(start_n, end_n + 1):
        print("starting problems with " + str(n) + "blocks")
        classical_times.append([])
        good_hint_times.append([])
        bad_hint_times.append([])
        while len(classical_times[-1]) < num_trials:
            print("  trial", len(classical_times[-1]))
            game.random_instance(n)
            # Iterative plan to determine optimal plan length
            planner = Z3Planner(game.problem, disp=False)
            plan = planner.iterative_plan()
            # Don't use random problems that are in goal state
            if (len(plan) == 0):
                continue
            steps = len(plan)
            hint_step = steps//2
            # step numbers are 1-indexed
            good_hint = (plan[hint_step -1], hint_step)
            # impossible to be in optimal plan:
            bad_hint = (("stack", game.problem.objects[-1], game.problem.objects[-2]), hint_step)
            gc.collect()

            classicalPlanner = Z3Planner(game.problem, steps, disp=False)
            classicalPlanner.smt_encode()
            start = time()
            classicalPlanner.check_sat()
            classical_times[-1].append(time() - start)
            gc.collect()

            goodHintPlanner = Z3Planner(game.problem, steps, disp=False)
            goodHintPlanner.smt_encode()
            start = time()
            goodHintPlanner.hint_plan([good_hint])
            # z3 push/pop take a lot of time relative to planning
            # time on small instances, so don't include in stats
            good_hint_times[-1].append(time() - start - goodHintPlanner.z3_stack_time)
            #print("good stack time", goodHintPlanner.z3_stack_time)
            if not goodHintPlanner.hint_used:
                pdb.set_trace()

            gc.collect()

            badHintPlanner = Z3Planner(game.problem, steps, disp=False)
            badHintPlanner.smt_encode()
            start = time()
            badHintPlanner.hint_plan([bad_hint])
            # z3 push/pop take a lot of time relative to planning
            # time on small instances, so don't include in stats
            bad_hint_times[-1].append(time() - start - badHintPlanner.z3_stack_time)
            #print("bad stack time", badHintPlanner.z3_stack_time)
            if badHintPlanner.hint_used:
                pdb.set_trace()

            gc.collect()

    print("classical times", classical_times)
    print("hint times", good_hint_times)
    print("bad times", bad_hint_times)

    subplots = plt.subplots(1, end_n-start_n + 1)
    fig, axs = subplots[0], subplots[1]
    fig.suptitle('Planner Performance')
    fig.supylabel("Time (seconds)")

    for ax_num in range(len(axs)):
        ax = axs[ax_num]

        values = [classical_times[ax_num], good_hint_times[ax_num], bad_hint_times[ax_num]]
        labels = ["classical", "good hint", "bad hint"]
        ax.boxplot(values)
        ax.set_xticklabels(labels)
        ax.set(xlabel=str(start_n + ax_num) + " blocks")
    plt.show()


if __name__ == "__main__":
    main()
