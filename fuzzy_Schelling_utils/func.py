import math
import random
import numpy as np
from scipy.stats import betabinom

def betabinom_mus(ALPHA, BETA, NUM_AGENTS):
    return betabinom.rvs(10, ALPHA, BETA, size=NUM_AGENTS) /10

def Moore_neighbors(row: int, col: int, NUM_ROW: int, NUM_COL: int) ->list:
    """function which returns Moore neighbors list of (row, col)"""
    if row == 0:
        row_under, row_self, row_upper = NUM_ROW - 1, row, row + 1
    elif row == NUM_ROW - 1:
        row_under, row_self, row_upper = row - 1, row, 0
    else:
        row_under, row_self, row_upper = row - 1, row, row + 1

    if col == 0:
        col_under, col_self, col_upper = NUM_COL - 1, col, col + 1
    elif col == NUM_COL - 1:
        col_under, col_self, col_upper = col - 1, col, 0
    else:
        col_under, col_self, col_upper = col - 1, col, col + 1

    Moore_neighbors = [(row, col) for row in [row_under, row_self, row_upper] for col in [col_under, col_self, col_upper]]
    Moore_neighbors.remove((row, col))
    return Moore_neighbors

class Agent:
    def __init__(self, row:int, col:int, mu:float, p:float):
        # row, col in {0,1,2,...}
        # mu: membership value to A, mu in [0,1], in actual trial {0, 0.1, 0.2, ..., 1}
        # p: percent similar wanted, p in [0,1]
        self.row, self.col = row, col
        self.mu = mu
        self.p = p
        self.neighbor_agents_mu = []
        self.s_sq = None
        self.fuzzy = None
        self.percent_wanted_neighbor = None
        self.next_move = False

    def search_neighbors(self, agents_field):
        # picking up Moore neighbors address of self
        moore_neighbors_address = Moore_neighbors(self.row, self.col, agents_field.shape[0], agents_field.shape[1])
        # neighbors agents list
        neighbor_agents = [agents_field[address] for address in moore_neighbors_address]
        # neighbors agents mus list
        self.neighbor_agents_mu = [agent.mu for agent in neighbor_agents if agent != None]
        # s_sq: average neighbors similarity
        if len(self.neighbor_agents_mu) == 0:
            self.s_sq = None
        else:
            self.s_sq = 1 - np.mean((np.array(self.neighbor_agents_mu) - self.mu) ** 2)
        # fuzzy: fuzziness of A in the neighborhoods
        neighbor_incl_myself_mu = self.neighbor_agents_mu + [self.mu]
        self.fuzzy = np.mean([H(x) for x in neighbor_incl_myself_mu])

    def moving_decision(self):
        if len(self.neighbor_agents_mu) == 0:
            self.percent_wanted_neighbor = 0
        else:
            if self.mu > 0.5:
                self.percent_wanted_neighbor = len([mu for mu in self.neighbor_agents_mu if mu >= self.mu]) / len(self.neighbor_agents_mu)
            elif self.mu < 0.5:
                self.percent_wanted_neighbor = len([mu for mu in self.neighbor_agents_mu if mu <= self.mu]) / len(self.neighbor_agents_mu)
            else:
                self.percent_wanted_neighbor = 1
        if self.p == 0:
            self.next_move = False
        elif self.percent_wanted_neighbor >= self.p:
            self.next_move = False
        else:
            self.next_move = True

def initial_field_address(NUM_ROW: int, NUM_COL: int, NUM_AGENTS: int):
    #overall address list
    field_address = [(x, y) for x in range(NUM_ROW) for y in range(NUM_COL)]
    # initial occupied address list
    occupied_address = random.sample(field_address, NUM_AGENTS)
    # non occupied address list
    non_occupied_address = list(set(field_address) - set(occupied_address))
    return occupied_address, non_occupied_address

def initializing_field(NUM_ROW: int, NUM_COL: int, dtype=object):
    """
    making 2 dimensional numpy array for storing agents' information
    """
    field = np.zeros([NUM_ROW, NUM_COL], dtype=dtype)
    field[:,:] = None # if dtype is float, then nan
    return field

def H(p: float) ->float:
    """
    binary entropy function. H: [0, 1] -> [0, 1]
    """
    if p == 0 or p == 1:
        return 0
    else:
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def D(p: float, q: float) ->float:
    """
    binary KL divergence
    """
    if p == 1:
        if q == 0:
            return math.inf
        else:
            return p * math.log2(p / q)
    elif p == 0:
        if q == 1:
            return math.inf
        else:
            return (1 - p) * math.log2((1 - p) / (1 - q))
    else:
        if q == 0 or q == 1:
            return math.inf
        else:
            return p * math.log2(p / q) + (1 - p) * math.log2((1 - p) / (1 - q))