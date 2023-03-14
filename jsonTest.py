import json

l = ['a', 0, False]
data = json.dumps(l)

print(json.loads(data))