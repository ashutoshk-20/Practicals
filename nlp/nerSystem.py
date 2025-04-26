import spacy
from sklearn.metrics import precision_score,recall_score,f1_score

nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text,ent.label_) for ent in doc.ents]

def evaluate_entities(true_entities, predicted_entities):
    y_true = [1 if entity in true_entities else 0 for entity in true_entities]
    y_pred = [1 if entity in predicted_entities else 0 for entity in predicted_entities]

    precision = precision_score(y_true,y_pred,zero_division=0)
    recall = recall_score(y_true,y_pred,zero_division=0)
    f1 = f1_score(y_true,y_pred,zero_division=0)

    return precision,recall,f1

user_input = input("Enter a sentence: ")
true_entities = [entity.strip() for entity in input("Enter true entities (comma separated): ").split(",")]

predicted_entities = extract_entities(user_input)

print("\nNamed Entities:")

for entity in predicted_entities:
    print(f"Entity: {entity[0]}, Label: {entity[1]}")

predicted_entities_text = [entity[0] for entity in predicted_entities]
precision, recall, f1 = evaluate_entities(true_entities, predicted_entities_text)

print(f"\nEvaluation Metrics:\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")