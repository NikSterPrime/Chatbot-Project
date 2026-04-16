# Chatbot with context memory project

## Structure of Project
```text
Chatbot_Project/
├── README.md
├── Documentation.md
├── data/
│   └── intent.json
└── src/
	├── chatbot.py
	├── main.py
	├── model.py
	├── preprocessing.py
	└── utils.py
```
- The project was structured keeping in mind readability and scalability.
- This also allows for easier debugging.


## Flow of the project
```text
src/main.py
	-> src/chatbot.py
	-> src/utils.py + data/intent.json
	-> src/preprocessing.py
	-> src/model.py
	-> predict_intent_details()
	-> src/chatbot.py
    ->Repeats until exited
```