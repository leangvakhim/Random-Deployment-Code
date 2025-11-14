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
dim = 10
lb = -100
ub = 100

value = ssapm(
    objective_function=benchmark_function,
    iter_max=iter_max,
    m_guilds=m_guilds,
    sparrow_per_guild=sparrow_per_guild,
    dim=dim,
    lb=lb,
    ub=ub,
)

print(f"Value is: {value}")