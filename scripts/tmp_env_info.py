import sys

print("PYTHON", sys.version)

try:
    import ray
    print("RAY", ray.__version__)
except Exception as e:
    print("RAY_IMPORT_ERROR", repr(e))

try:
    import grpc
    print("GRPC", grpc.__version__)
except Exception as e:
    print("GRPC_IMPORT_ERROR", repr(e))
