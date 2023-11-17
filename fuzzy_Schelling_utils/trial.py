import math
import random
import numpy as np
import fuzzy_Schelling_utils.func as fs

def trial(NUM_ROW, NUM_COL, DENSITY, PERCENT_SIMILAR_WANTED, NUM_TRIALS, ALPHA, BETA):
    NUM_AGENTS = int((NUM_ROW * NUM_COL) * DENSITY // 1)

    # random agents' mu list from random betabinomial distribution
    agent_mus = fs.betabinom_mus(ALPHA, BETA, NUM_AGENTS)

    # initial field address
    occupied_address, non_occupied_address = fs.initial_field_address(NUM_ROW, NUM_COL, NUM_AGENTS)

    # assigning agent instances to field
    agents_field = fs.initializing_field(NUM_ROW, NUM_COL)
    next_move_address = []
    unhappy_mu_list = []
    similarity_list = []
    fuzzy_list = []
    for i, address in enumerate(occupied_address):
        agents_field[address] = fs.Agent(address[0], address[1], agent_mus[i], PERCENT_SIMILAR_WANTED)
        # moving decision for each agent
        agents_field[address].search_neighbors(agents_field)
        agents_field[address].moving_decision()
        similarity_list.append(agents_field[address].s_sq)
        fuzzy_list.append(agents_field[address].fuzzy)
        if agents_field[address].next_move:
            next_move_address.append(address)
            unhappy_mu_list.append(agents_field[address].mu)

    unhappy_percent_trend = []
    unhappy_percent_trend.append(len(unhappy_mu_list) / NUM_AGENTS)
    average_similarity_trend = []
    average_similarity_trend.append(np.nanmean(np.array(similarity_list, dtype=float)))
    average_fuzzy_trend = []
    average_fuzzy_trend.append(np.nanmean(np.array(fuzzy_list, dtype=float)))

    for t in range(1, NUM_TRIALS + 1):
        # moving process
        random.shuffle(next_move_address)
        random.shuffle(non_occupied_address)
        for address in next_move_address:
            occupied_address.remove(address)
            next_address = non_occupied_address.pop(0)
            # cross assigning agent instances
            agents_field[next_address], agents_field[address] = agents_field[address], agents_field[next_address]
            agents_field[next_address].row = next_address[0]
            agents_field[next_address].col = next_address[1]
            occupied_address.append(next_address)
            non_occupied_address.append(address)

        # moving decision for each agent
        next_move_address = []
        unhappy_mu_list = []
        similarity_list = []
        fuzzy_list = []
        for address in occupied_address:
            agents_field[address].search_neighbors(agents_field)
            agents_field[address].moving_decision()
            similarity_list.append(agents_field[address].s_sq)
            fuzzy_list.append(agents_field[address].fuzzy)
            if agents_field[address].next_move:
                next_move_address.append(address)
                unhappy_mu_list.append(agents_field[address].mu)

        unhappy_percent_trend.append(len(unhappy_mu_list) / NUM_AGENTS)
        average_similarity_trend.append(np.nanmean(np.array(similarity_list, dtype=float)))
        average_fuzzy_trend.append(np.nanmean(np.array(fuzzy_list, dtype=float)))

        # if everyone happy, then break
        if not next_move_address:
            break

    return unhappy_percent_trend, average_similarity_trend, average_fuzzy_trend



def trial_simple_Schelling(NUM_ROW, NUM_COL, DENSITY, PERCENT_SIMILAR_WANTED, NUM_TRIALS):
    NUM_AGENTS = int((NUM_ROW * NUM_COL) * DENSITY // 1)

    # random agents' mu list from [0,1]
    agent_mus = [random.randint(0,1) for _ in range(NUM_AGENTS)]

    # initial field address
    occupied_address, non_occupied_address = fs.initial_field_address(NUM_ROW, NUM_COL, NUM_AGENTS)

    # assigning agent instances to field
    agents_field = fs.initializing_field(NUM_ROW, NUM_COL)
    next_move_address = []
    unhappy_mu_list = []
    similarity_list = []
    fuzzy_list = []
    for i, address in enumerate(occupied_address):
        agents_field[address] = fs.Agent(address[0], address[1], agent_mus[i], PERCENT_SIMILAR_WANTED)
        # moving decision for each agent
        agents_field[address].search_neighbors(agents_field)
        agents_field[address].moving_decision()
        similarity_list.append(agents_field[address].s_sq)
        fuzzy_list.append(agents_field[address].fuzzy)
        if agents_field[address].next_move:
            next_move_address.append(address)
            unhappy_mu_list.append(agents_field[address].mu)

    unhappy_percent_trend = []
    unhappy_percent_trend.append(len(unhappy_mu_list) / NUM_AGENTS)
    average_similarity_trend = []
    average_similarity_trend.append(np.nanmean(np.array(similarity_list, dtype=float)))
    average_fuzzy_trend = []
    average_fuzzy_trend.append(np.nanmean(np.array(fuzzy_list, dtype=float)))

    for t in range(1, NUM_TRIALS + 1):
        # moving process
        random.shuffle(next_move_address)
        random.shuffle(non_occupied_address)
        for address in next_move_address:
            occupied_address.remove(address)
            next_address = non_occupied_address.pop(0)
            # cross assigning agent instances
            agents_field[next_address], agents_field[address] = agents_field[address], agents_field[next_address]
            agents_field[next_address].row = next_address[0]
            agents_field[next_address].col = next_address[1]
            occupied_address.append(next_address)
            non_occupied_address.append(address)

        # moving decision for each agent
        next_move_address = []
        unhappy_mu_list = []
        similarity_list = []
        fuzzy_list = []
        for address in occupied_address:
            agents_field[address].search_neighbors(agents_field)
            agents_field[address].moving_decision()
            similarity_list.append(agents_field[address].s_sq)
            fuzzy_list.append(agents_field[address].fuzzy)
            if agents_field[address].next_move:
                next_move_address.append(address)
                unhappy_mu_list.append(agents_field[address].mu)

        unhappy_percent_trend.append(len(unhappy_mu_list) / NUM_AGENTS)
        average_similarity_trend.append(np.nanmean(np.array(similarity_list, dtype=float)))
        average_fuzzy_trend.append(np.nanmean(np.array(fuzzy_list, dtype=float)))

        # if everyone happy, then break
        if not next_move_address:
            break

    return unhappy_percent_trend, average_similarity_trend, average_fuzzy_trend