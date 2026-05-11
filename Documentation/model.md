# Model.py

## Dependencies
1. `numpy`
2. `TfidfVectorizer` from `sklearn.feature_extraction.text`
3. `LogisticRegression` from `sklearn.linear_model`
4. `exact_pattern_to_tag`,`labels`,`texts` from `utils()`
5. `clean_text` from `preprocessing()`

## Inputs
1. `texts`: List of input texts from `k_intents.json`
2. `labels`: List of corresponding labels for each pattern
3. `exact_pattern_to_tag`: Dictionary mapping cleaned pattern string to their intent for instant lookup
4. `text`: The input text to classify that is taken during runtime.

## Outputs
1. `predict_intent_details()`: returns `intent`,`confidence`,`tag`
2. `predict_intent_rankings()`: returns top 3(`top_k`) in list of (`intent`,`score`) tuple.
3. `predict_intent_with_confidence()`: returns `intent` and `confidence` for the input text.
