import flax
import msgpack
import sys

with open("checkpoints/checkpoint_20260706-1701/model.msgpack", "rb") as f:
    data = flax.serialization.msgpack_restore(f.read())

print("Keys in model.msgpack:")
if isinstance(data, dict):
    print(data.keys())
    for k in data.keys():
        if isinstance(data[k], dict):
            print(f"  {k} keys: {data[k].keys()}")
else:
    print(type(data))
