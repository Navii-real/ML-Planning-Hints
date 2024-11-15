# Utils for handling PDDL
# Uses python built-in tuples to represent syntax tree

import re
import pdb
import os
import numpy as np

class Parser:
    ATOM_RE = re.compile("^[^()\s]+$")
    LIST_RE = re.compile("^\(.+\)$", flags=re.DOTALL)

    @classmethod
    def split_list(cls, s):
        start, depth, out = 0, 0, []
        for i in range(len(s)):
            char = s[i]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    out.append(s[start: i+1])
                    start = i + 1
                elif depth < 0:
                    raise Exception("invalid ) at index " + str(i) + ' in "' + s + '"')
            elif depth == 0 and re.match("\s", char):
                if i > start:
                    out.append(s[start: i])
                start = i + 1
        if start < len(s):
            out.append(s[start:len(s)])
        return out

    @classmethod
    def parse(cls, s):
        if cls.ATOM_RE.match(s):
            return s
        elif cls.LIST_RE.match(s):
            return tuple(map(cls.parse, cls.split_list(s[1:-1])))
        else:
            raise Exception('invalid pddl string: "' + s + '"')

    @classmethod
    def read_from_file(cls, fname):
        assert(os.path.exists(fname))
        f = open(fname, 'r')
        s = f.read().strip()
        f.close()
        return cls.parse(s)

# TODO move RE stuff from parser to expression?
class Expression:
    @classmethod
    def sub(cls, expr, old, new):
        if type(expr) is str:
            if expr == old:
                return new
            return expr
        if type(expr) is tuple:
            return tuple([cls.sub(term, old, new) for term in expr])
        raise Exception("invalid expression")

class Domain:
    @classmethod
    def read_from_file(cls, fname):
        return cls(Parser.read_from_file(fname))

    def __init__(self, expr):
        if not type(expr) is tuple:
            raise Exception("pddl expression must be a tuple")
        if len(expr) == 0 or expr[0] != "define":
            raise Exception("Top level pddl expression must be a 'define'")
        self.name = None
        self.predicates = {}
        self.actions = {}
        for clause in expr[1:]:
            self.add_definition(clause)
        if self.name is None:
            raise Exception("Domain definition must have a 'domain' clause")
        if len(self.predicates) == 0:
            raise Exception("Domain definition must have predicates")
        if len(self.actions) == 0:
            raise Exception("Domain definition must have actions")

    def add_definition(self, clause):
        if not type(clause) is tuple or len(clause) < 1:
            raise Exception("define clauses must be tuples with length >=1")
        if clause[0] == "domain":
            self.add_domain_clause(clause)
        elif clause[0] == ":predicates":
            self.add_predicates_clause(clause)
        elif clause[0] == ":action":
            self.add_action_clause(clause)

    def add_domain_clause(self, clause):
        if not self.name is None:
            raise Exception("got domain clause but domain already defined")
        if len(clause) != 2:
            raise Exception("domain clause must have two elements")
        self.name = clause[1]

    def add_predicates_clause(self, clause):
        for definition in clause[1:]:
            pred = Predicate.from_pddl(definition)
            if pred.name in self.predicates:
                raise Exception('Predicate "' + pred.name + '" already defined')
            self.predicates[pred.name] = pred

    def add_action_clause(self, clause):
        action = Action.from_pddl(clause)
        if action.name in self.actions:
            raise Exception('Action "' + action.name + '" already defined')
        self.actions[action.name] = action

    def __repr__(self):
        s = 'Domain "' + self.name + '"\nPREDICATES:\n'
        for pred in self.predicates.values():
            s = s + str(pred) + "\n"
        s = s + "ACTIONS:\n"
        for action in self.actions.values():
            s = s + str(action) + "\n"
        return s

class Predicate:
    @classmethod
    def from_pddl(cls, clause):
        if not type(clause) is tuple or len(clause) < 1:
            raise Exception("predicate definition clause must be a tuple with length >= 1")
        if not type(clause[0]) is str:
            raise Exception("predicate name must be a string")
        return cls(clause[0], len(clause) - 1)

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __repr__(self):
        return 'Predicate "' + self.name + '" with arity ' + str(self.arity)

class Action:
    @classmethod
    def from_pddl(cls, clause):
        if not type(clause) is tuple or len(clause) != 8:
            raise Exception("action definition clause must be a tuple with length 8")
        if not type(clause[1]) is str:
            raise Exception("action name must be a string")
        name, params, pre, eff = clause[1], None, None, None
        for i in [2, 4, 6]:
            if clause[i] == ":parameters":
                params = clause[i+1]
            elif clause[i] == ":precondition":
                if not type(clause[i+1]) is tuple or clause[i+1][0] != "and":
                    raise Exception("precondition must be a conjunction")
                pre = clause[i+1][1:] # TODO logical expr parsing
            elif clause[i] == ":effect":
                if not type(clause[i+1]) is tuple or clause[i+1][0] != "and":
                    raise Exception("effect must be a conjunction")
                eff = clause[i+1][1:] # TODO logical expr parsing
            else:
                raise Exception("unexpected action attribute: " + str(clause[i]))
        return cls(name, params, pre, eff)

    def __init__(self, name, params, pre, eff):
        if None in [name, params, pre, eff]:
            raise Exception("action requires name, parameters, precondition, and effect")
        self.name = name
        self.params = params
        self.arity = len(params)
        self.pre = pre
        self.eff = eff

    def __repr__(self):
        return 'Action "' + self.name + '" | params: ' + str(self.params) \
            + " | pre: " + str(self.pre) + " | eff: " + str(self.eff)

class GroundPredicate:
    def __init__(self, pred, args):
        self.pred = pred
        self.args = args
        if len(args) != pred.arity:
            raise Exception("got " + str(len(args)) + " args for predicate with arity " + str(pred.arity))

    def expr(self):
        return (self.pred.name,) + self.args

    def __repr__(self):
        return str(self.expr())

class GroundAction:
    def __init__(self, action, args):
        self.action = action
        self.args = args
        if len(args) != len(action.params):
            raise Exception("got " + str(len(args)) + " args for action with arity " + str(len(action.params)))
        self.pre, self.eff = action.pre, action.eff
        for i in range(len(args)):
            self.pre = Expression.sub(self.pre, action.params[i], args[i])
            self.eff = Expression.sub(self.eff, action.params[i], args[i])

    def expr(self):
        return (self.action.name,) + self.args

    def __repr__(self):
        return str(self.expr())

class Problem:
    # iterator of tuples of objects
    class ObjectTupleIterator:
        def __init__(self, objects, n):
            self.objects = objects
            self.nobjs = len(objects)
            self.n = n
            self.i = 0

        def __iter__(self):
            return self

        def __next__(self):
            # special case
            if (self.n == 0):
                if self.i == 0:
                    self.i += 1
                    return ()
                else:
                    raise StopIteration
            indices = []
            while len(np.unique(indices)) < self.n:
                if self.i >= self.nobjs**self.n:
                    raise StopIteration
                indices = [self.i % (self.nobjs ** (p+1)) // (self.nobjs ** p)
                           for p in range(self.n)]
                self.i += 1
            return tuple([self.objects[ind] for ind in indices])

    @classmethod
    def read_from_file(cls, domain, fname):
        return cls.from_pddl(domain, Parser.read_from_file(fname))

    @classmethod
    def from_pddl(cls, domain, expr):
        if not type(domain) is Domain:
            raise Exception("Problem domain must be a domain")
        if not type(expr) is tuple or len(expr) != 6:
            raise Exception("Problem pddl must be a tuple with length 6")
        if expr[0] != "define":
            raise Exception("Top level pddl expression must be a 'define'")
        name, objects, init, goal = None, None, None, None
        for clause in expr[1:]:
            if not type(clause) is tuple or len(clause) < 2:
                raise Exception("Problem definition clauses must be tuples with length >= 2")
            if clause[0] == "problem":
                if len(clause) > 2:
                    raise Exception("Problem clause must have length 2")
                name = clause[1]
            elif clause[0] == ":domain":
                if len(clause) > 2:
                    raise Exception("domain clause must have length 2")
                if clause[1] != domain.name:
                    raise Exception('problem is defined for domain "' + str(clause[1]) \
                                    + '" which does not match the given domain object with name "' \
                                    + str(domain.name) + '"')
            elif clause[0] == ":objects":
                objects = clause[1:]
            elif clause[0] == ":init":
                init = clause[1:]
            elif clause[0] == ":goal":
                if len(clause) > 2:
                    raise Exception("Goal clause must have length 2")
                if not type(clause[1]) is tuple or clause[1][0] != "and":
                    raise Exception("Goal must be a conjunction")
                goal = clause[1][1:]
            else:
                raise Exception("unexpected problem attribute: " + str(clause[0]))
        return cls(name, domain, objects, init, goal)

    def __init__(self, name, domain, objects, init, goal):
        if None in [name, domain, objects, init, goal]:
            raise Exception("Problem requires name, domain, objects, initial state, and goal")
        self.name = name
        self.domain = domain
        self.objects = objects
        self.init = init
        self.goal = goal

    def ground_iterator(self, n):
        return Problem.ObjectTupleIterator(self.objects, n)

    def generate_ground_actions(self):
        ground = []
        for action in self.domain.actions.values():
            for args in self.ground_iterator(action.arity):
                ground.append(GroundAction(action, args))
        return ground

    def generate_ground_predicates(self):
        ground = []
        for pred in self.domain.predicates.values():
            for args in self.ground_iterator(pred.arity):
                ground.append(GroundPredicate(pred, args))
        return ground

    def __repr__(self):
        return 'Problem "' + self.name + '" in domain "' + self.domain.name + '"\nOBJECTS: ' \
            + str(self.objects) + "\nINITIAL STATE: " + str(self.init) \
            + "\nGOAL: " + str(self.goal)
