from utils.hf_model_localizer import local_load_hf_sentiment, local_load_hf_zeroshotclassification
from transformers import pipeline

# Importing the sentiment model via hf_model_localizer
# model, tokenizer = local_load_hf_sentiment()
# Importing the zero-shot-classification model via hf_model_localizer
model, tokenizer = local_load_hf_zeroshotclassification()

# # Running the classification task
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# message = classifier(
#     ["I just hate that stuff.",
#     "This is so great!",
#     "I am feeling neutral about this."]
#     )

# print(message)


classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
message = classifier(
    "This is a course about inflation",
    candidate_labels=["education", "politics", "economy"],
)

print(message)

