from transformers import pipeline

classifier = pipeline('sentiment-analysis')
classifier("I just hate that stuff.")

print(classifier)
