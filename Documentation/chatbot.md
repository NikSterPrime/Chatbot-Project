# Chatbot.py

## Imports in the files
1. `os`
2. `random`
3. `re`
4. `json`
5. `deque` from `collections`
6. `datetime` from `datetime`
7. `Path` from `pathlib`
8. `predict_intent_details`, `predict_intent_rankings` from `model`
9. `intent_to_responses` from `utils`
10. `is_podcast_request`,`recommend_podcast_from_queue` from `podcast_recommender`


## What problem does Chatbot.py solve?

- Chatbot.py is the main file that interacts and run the main chatbot logic which handles user input.
- This file also concentrates on setting up Gemini Integration which is used for fallback and chat summaries. Gemini Integration is setup using the 'gemini-2.5-flash' model with the assistance of langchain.
- Another problem solves or most like feature it adds is the ability to store session memory for the user for that session, so that it can use the context to provide more accurate responses.
- It adds the feature of podcast recommendation based on the user's input and preferences.
- This file also is used to render help to the user who is unfimiliar with commands. eg: access memory need the command `\memory`, this unknown to the user the funtion `_render_help()` is used to render the help message.
- The file also defines `CONFIDENCE_THRESHOLD`,`MARGIN_THRESHOLD`,`PERSISTENT_MEMORY_PATH` where `CONFIDENCE_THRESHOLD` is the confidence threshold for the model to consider a response as valid, `MARGIN_THRESHOLD` is the margin threshold for the model to consider a response as valid, and `PERSISTENT_MEMORY_PATH` is the path to the persistent memory file where session memory is stored.
