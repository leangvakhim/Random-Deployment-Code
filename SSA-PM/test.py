import numpy as np
from benchmark import (
    sphere_function,
)
from ssapm import (
    ssapm,
)

benchmark_function = sphere_function
iter_max = 100
m_guilds = 4
sparrow_per_guild = 25
dim = 30
lb = -100
ub = 100
# values_list = []

for _ in range(10):
    value = ssapm(
        objective_function=benchmark_function,
        iter_max=iter_max,
        m_guilds=m_guilds,
        sparrow_per_guild=sparrow_per_guild,
        dim=dim,
        lb=lb,
        ub=ub,
    )
    # values_list.append(value)

# print(f"Mean of the fitness values: {np.mean(values_list)}")

# print(f"Value is: {value}")