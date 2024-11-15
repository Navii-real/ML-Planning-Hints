import z3
from time import time

class Z3Planner:
    def __init__(self, problem, horizon=0, disp = True):
        if (disp):
            print("initializing problem...")
        self.domain = problem.domain
        self.problem = problem
        self.horizon = horizon
        self.solver = z3.Solver()
        self.vars = {}
        self.ground_preds = self.problem.generate_ground_predicates()
        self.ground_actions = self.problem.generate_ground_actions()
        self.generate_pred_changers()
        self.sat = None
        self.iterative = False
        self.disp = disp
        if disp:
            print(" done")

    def generate_pred_changers(self):
        self.adders = {}
        self.deleters = {}
        for action in self.ground_actions:
            for clause in action.eff:
                if clause[0] == 'not':
                    pred = clause[1]
                    if pred in self.deleters:
                        self.deleters[pred].append(action.expr())
                    else:
                        self.deleters[pred] = [action.expr()]
                else:
                    if clause in self.adders:
                        self.adders[clause].append(action.expr())
                    else:
                        self.adders[clause] = [action.expr()]

    def add_var(self, name):
        if name in self.vars:
            raise Exception("variable '" + name + "' is already defined")
        self.vars[name] = z3.Bool(name)

    def var_name(self, expr, step):
        return "_".join(expr + (str(step),))

    def get_var(self, expr, step):
        return self.vars[self.var_name(expr, step)]

    def smt_add_step_vars(self, step):
        for pred in self.ground_preds:
            self.add_var(self.var_name(pred.expr(), step))
        if (step > 0):
            for action in self.ground_actions:
                self.add_var(self.var_name(action.expr(), step))

    def smt_encode_init(self):
        self.smt_add_step_vars(0)
        init_t = [self.get_var(clause, 0) for clause in self.problem.init]
        init_f = [z3.Not(self.get_var(pred.expr(), 0)) for pred in self.ground_preds if not pred.expr() in self.problem.init]
        self.solver.add(z3.And(*(init_t + init_f)))

    def smt_encode_ops(self, step):
        for action in self.ground_actions:
            pre = [self.get_var(pred, step - 1)
                   for pred in action.pre]
            eff_pos = [self.get_var(clause, step)
                       for clause in action.eff if clause[0] != "not"]
            eff_neg = [z3.Not(self.get_var(clause[1], step))
                       for clause in action.eff if clause[0] == "not"]
            self.solver.add(
                z3.Implies(self.get_var(action.expr(), step),
                           z3.And(*(pre + eff_pos + eff_neg))))
    def smt_encode_op_mutex(self, step):
        self.solver.add(z3.Not(z3.Or(
            *[z3.And(self.get_var(op1.expr(), step),
                     self.get_var(op2.expr(), step))
              for op1 in self.ground_actions
              for op2 in self.ground_actions if op1 != op2])))

    def smt_encode_frame_axioms(self, step):
        for pred in self.ground_preds:
            prev = self.get_var(pred.expr(), step - 1)
            curr = self.get_var(pred.expr(), step)
            adders = [self.get_var(a, step) for a in self.adders[pred.expr()]]
            deleters = [self.get_var(a, step) for a in self.deleters[pred.expr()]]
            self.solver.add(z3.Or(prev == curr, *(adders + deleters)))

    def smt_encode_step(self, step):
        self.smt_add_step_vars(step)
        self.smt_encode_ops(step)
        self.smt_encode_op_mutex(step)
        self.smt_encode_frame_axioms(step)

    def smt_encode(self):
        if self.disp:
            print("Encoding problem...")
        self.smt_encode_init()
        for step in range(1, self.horizon + 1):
            self.smt_encode_step(step)
        if self.disp:
            print(" done.")

    def check_sat(self):
        if (self.iterative):
            self.solver.push()
        else:
            if self.disp:
                print("Planning...")
        for clause in self.problem.goal:
            self.solver.add(self.get_var(clause, self.horizon))
        self.sat = self.solver.check()
        if self.disp:
            print(self.sat)

    def get_plan(self):
        if self.sat is None:
            raise Exception("Must check model before extracting plan")
        if str(self.sat) == "sat":
            m = self.solver.model()
            plan = []
            for i in range(1, self.horizon + 1):
                actions = [a for a in self.ground_actions
                           if m[self.get_var(a.expr(), i)]]
                if len(actions) > 0:
                    plan.append(actions[0].expr())
            return tuple(plan)

        else:
            raise Exception("Can't get plan because problem is unsat")

    def plan(self):
        self.smt_encode()
        self.check_sat()

        if str(self.sat) == "sat":
            return self.get_plan()
        else:
            return None

    def iterative_plan(self):
        if self.horizon != 0:
            raise Exception("Must plan from step 0!")
        self.iterative = True
        self.smt_encode_init()
        # Should not go over this limit for blocksworld
        for step in range(1, 4 * len(self.problem.objects)):
            if self.disp:
                print("trying step...", step)
            self.horizon = step
            self.smt_encode_step(step)
            self.check_sat()
            if str(self.sat) == "sat":
                return self.get_plan()
            else:
                self.solver.pop()

    # Takes tuple of (action expression, time_step)
    # requires planner to already be encoded with self.smt_encode()
    def hint_plan(self, hints):
        # need to track z3 push/pop time for stats
        start = time()
        self.solver.push()
        self.z3_stack_time = time() - start
        self.hint_used = True

        for hint in hints:
            self.solver.add(self.get_var(hint[0], hint[1]))
        self.check_sat()
        
        if str(self.sat) == "sat":
            if self.disp:
                print("Found plan with hint!")
            return self.get_plan()
        else:
            if self.disp:
                print("Could not satisfy hints. Trying without hints.")
        self.hint_used = False
        start = time()
        self.solver.pop()
        self.z3_stack_time += time() - start
        self.check_sat()
        if str(self.sat) == "sat":
            if self.disp:
                print("Found plan without hints.")
            return self.get_plan()
        else:
            if self.disp:
                print("Could not find plan")
            return None
