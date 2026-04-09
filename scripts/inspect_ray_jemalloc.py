import ray._private.ray_constants as rc

names = [n for n in dir(rc) if "JEMALLOC" in n.upper() or "PRELOAD" in n.upper()]
print(names)
for n in names:
    print(n, getattr(rc, n))
