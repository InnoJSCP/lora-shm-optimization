import numpy as np
from deap import base, creator, tools, algorithms
import random
from src.lora_phy import energy_tx
from src.cs_utils import generate_sensing_matrix, compress, recover_bp, quality_metric

SF_VALUES = [7,8,9,10,11,12]
CR_VALUES = [4/5, 4/6, 4/7, 4/8]
CF_VALUES = np.linspace(0.1, 1.0, 20)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate(individual, signal, noise_snr_db, channel_type='AWGN', k_factor=3):
    sf_idx, cr_idx, cf_idx = individual
    sf = SF_VALUES[sf_idx]
    cr = CR_VALUES[cr_idx]
    cf = CF_VALUES[cf_idx]
    N = len(signal)
    M = max(1, int(cf * N))
    Phi = generate_sensing_matrix(M, N, seed=random.randint(0, 100000))
    y = compress(signal, Phi)
    sigma = np.std(y) * 10 ** (-noise_snr_db / 20)
    y_noisy = y + np.random.randn(M) * sigma
    x_hat = recover_bp(y_noisy, Phi, epsilon=0.1*sigma)
    q = quality_metric(signal, x_hat)
    payload_bytes = int(np.ceil(M / 8))
    e = energy_tx(sf, cr, payload_bytes)
    return e, -q

def setup_toolbox(signal, noise_snr_db):
    toolbox = base.Toolbox()
    toolbox.register("attr_sf", random.randint, 0, len(SF_VALUES)-1)
    toolbox.register("attr_cr", random.randint, 0, len(CR_VALUES)-1)
    toolbox.register("attr_cf", random.randint, 0, len(CF_VALUES)-1)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_sf, toolbox.attr_cr, toolbox.attr_cf), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, signal=signal, noise_snr_db=noise_snr_db)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[0,0,0], up=[len(SF_VALUES)-1, len(CR_VALUES)-1, len(CF_VALUES)-1], indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def run_nsga2(signal, noise_snr_db, pop_size=100, ngen=200):
    toolbox = setup_toolbox(signal, noise_snr_db)
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size,
                                         cxpb=0.9, mutpb=0.1, ngen=ngen,
                                         stats=stats, halloffame=hof, verbose=True)
    front = [(ind.fitness.values[0], -ind.fitness.values[1]) for ind in hof]
    return front, pop, log