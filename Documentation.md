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

## Documentation on functionality of each file

### main.py
The main python file that is run is used to call the `chatbot()` from src/chatbot.py
This can be used for further expansion/extension of the chatbot or its integration

### chatbot.py
Has two major functions - 
- `give_response()`:
    The function receives the intent that is predicted by the model.
    When `predict_intent_details()` is called it also creates an `dict:intent_to_response` a fast look up table so that the it doesnt need to look through the whole dataset.
    This create a table with the intent predicted along with the fallback intent.
    It then returns a random sentence for the intent.

- `chatbot()`:
    This is the main function that is run to interact with the chatbot. Takes user input (`var:user_input`)
    `user_input` is used as an argument for the `predict_intent_details()` imported from src/model.py which returns the intent, confidence and margin
    `var:intent`: predict label (like greeting, fallback, help).
    `var:confidence`:the probability of the highest intent among all the intent.
    `var:margin`: tell the difference between highest probable intent and second highest probable intent.
    We setup a treshold for confidence and margin where the the returned values are compared to it, the threshold is initialised before running if the value is lesser than threshold it changes intent to a `FALLBACK_TAG`
    It then calls the `give_response()` with `var:intent` as an argument.
    This continues until user gives the exit command.


### model.py
This file consists of the model and vectorisation of the training data and user input