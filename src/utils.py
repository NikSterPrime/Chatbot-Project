import json

with open("/data/intent.json") as file:
    data = json.load(file)

#for intents in data["intents"]:
#   print(intents)

text = []
label = []

for intents in data["intents"]:
    for pattern in intents["patterns"]:
        text.append(pattern)
        label.append(intents["tag"])

