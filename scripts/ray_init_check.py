import ray

ray.init(address="auto")
print(ray.cluster_resources())
