# Learning Hints for Automated Planning
### Copied over from old github account
#### Deep reinforcement learning to improve performance of a classical planner by generating "hints," or actions that are likely part of a valid plan.

## Environment

It is recommended but not required to install the required python modules (see requirements.txt) in a virtual environment.
If the instructions in the "Running" section do not work for you, try the following steps (All from the base directory of the project)

Leave any current virtual python environment you are in, e.g.
```
deactivate
```
or
```
conda deactivate
```
Create an empty virtual environment
```
python3 -m venv env
```
Enter the virtual environment
```
source env/bin/activate
```
Install the required libraries
```
pip3 install -r requirements.txt
```
When you are done working with the project, leave the virtual environment
```
deactivate
```
## Learning
Referring to the main_ql.py file, to learn for a given problem, start a game, and then
call the ql_learn_init() function. Modifying hyperparameters can be done within that function definition, which is located within utils/blocksworld_game.py. This function will generate 3 files
within the currect directory: q_table.npy, q_mask.npy, and state_index. These correspond to the 
Q table, the mask, and the state indexing dictionary that allow us to know if we've beem to a state
before respectively.

To test the learner, make sure the three files generated are in the current directory. Then, create
a game, making sure the number of blocks matches the number of blocks learned with. And then run 
test_q (refer again to main_ql.py). This will launch a command line game version of BlocksWorld
that uses the three generated files to make decisions about how to plan. Press enter to advance to the
next decision.
