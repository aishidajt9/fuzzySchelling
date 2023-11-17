import math
import random
import numpy as np
import fuzzy_Schelling_utils.func as fs

def trial_for_h(NUM_ROW, NUM_COL, DENSITY, PERCENT_SIMILAR_WANTED, NUM_TRIALS, ALPHA, BETA):
    NUM_AGENTS = int((NUM_ROW * NUM_COL) * DENSITY // 1)

    # random agents' mu list from random betabinomial distribution
    agent_mus = fs.betabinom_mus(ALPHA, BETA, NUM_AGENTS)

    # initial field address
    occupied_address, non_occupied_address = fs.initial_field_address(NUM_ROW, NUM_COL, NUM_AGENTS)

    # assigning agent instances to field
    agents_field = fs.initializing_field(NUM_ROW, NUM_COL)
    next_move_address = []
    mu_list = []
    sim_list = []
    fuzzy_list = []
    unhappy_mu_list = []
    for i, address in enumerate(occupied_address):
        agents_field[address] = fs.Agent(address[0], address[1], agent_mus[i], PERCENT_SIMILAR_WANTED)
        # moving decision for each agent
        agents_field[address].search_neighbors(agents_field)
        agents_field[address].moving_decision()
        mu_list.append(agents_field[address].mu)
        sim_list.append(agents_field[address].s_sq)
        fuzzy_list.append(agents_field[address].fuzzy)
        if agents_field[address].next_move:
            next_move_address.append(address)
            unhappy_mu_list.append(agents_field[address].mu)

    unhappy_percent_trend = []
    average_similarity_trend = []
    average_fuzzy_trend = []

    mus = np.arange(0, 0.6, 0.1)# 0.0 & 1.0, 0.1 & 0.9, 0.2 & 0.8, 0.3 & 0.7, 0.4 & 0.6, 0.5

    unhappy_percent = []
    average_similarity = []
    average_fuzzy = []
    for mu in mus:
        mu_size = sum(np.isclose(mu_list, mu) | np.isclose(mu_list, 1 - mu))
        unhappy_mu_size = sum(np.isclose(unhappy_mu_list, mu) | np.isclose(unhappy_mu_list, 1 - mu))
        unhappy_percent.append(unhappy_mu_size / mu_size)
        similarity = np.array(sim_list, dtype=float)[np.where(np.isclose(mu_list, mu) | np.isclose(mu_list, 1 - mu))]
        average_similarity.append(np.nanmean(similarity))
        fuzziness = np.array(fuzzy_list, dtype=float)[np.where(np.isclose(mu_list, mu) | np.isclose(mu_list, 1 - mu))]
        average_fuzzy.append(np.mean(fuzziness))

    unhappy_percent_trend.append(unhappy_percent)
    average_similarity_trend.append(average_similarity)
    average_fuzzy_trend.append(average_fuzzy)

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
        mu_list = []
        sim_list = []
        fuzzy_list = []
        unhappy_mu_list = []
        for address in occupied_address:
            agents_field[address].search_neighbors(agents_field)
            agents_field[address].moving_decision()
            mu_list.append(agents_field[address].mu)
            sim_list.append(agents_field[address].s_sq)
            fuzzy_list.append(agents_field[address].fuzzy)
            if agents_field[address].next_move:
                next_move_address.append(address)
                unhappy_mu_list.append(agents_field[address].mu)

        mus = np.arange(0, 0.6, 0.1)# 0.0 & 1.0, 0.1 & 0.9, 0.2 & 0.8, 0.3 & 0.7, 0.4 & 0.6, 0.5

        unhappy_percent = []
        average_similarity = []
        average_fuzzy = []
        for mu in mus:
            mu_size = sum(np.isclose(mu_list, mu) | np.isclose(mu_list, 1 - mu))
            unhappy_mu_size = sum(np.isclose(unhappy_mu_list, mu) | np.isclose(unhappy_mu_list, 1 - mu))
            unhappy_percent.append(unhappy_mu_size / mu_size)
            similarity = np.array(sim_list, dtype=float)[np.where(np.isclose(mu_list, mu) | np.isclose(mu_list, 1 - mu))]
            average_similarity.append(np.nanmean(similarity))
            fuzziness = np.array(fuzzy_list, dtype=float)[np.where(np.isclose(mu_list, mu) | np.isclose(mu_list, 1 - mu))]
            average_fuzzy.append(np.mean(fuzziness))

        unhappy_percent_trend.append(unhappy_percent)
        average_similarity_trend.append(average_similarity)
        average_fuzzy_trend.append(average_fuzzy)

        # if everyone happy, then break
        if not next_move_address:
            break

    return unhappy_percent_trend, average_similarity_trend, average_fuzzy_trend



