# NLI with NLP for Zero-Shot-Classification (ie untrained)
# facebook/bart-large-mnli model (most popular by downloads)
# https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Define the desired order of labels
fixed_labels_order = ["product release", "in-person event", "company announcement"]
results = classifier([
    "Give the release notes of the winter '24 release.",
    "Is Copado planning to go public (IPO) this year?",
    "Will Copado be attending Dreamforce in SF?",
    ], candidate_labels=fixed_labels_order)
for result in results:
    # Create a dictionary to map labels to scores
    scores_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
    # Reorder and round the scores based on the fixed labels order
    ordered_rounded_scores = [round(scores_dict[label], 2) for label in fixed_labels_order]
    print(f"labels: {fixed_labels_order}, with scores: {ordered_rounded_scores}")
# labels: ['product release', 'in-person event', 'company announcement'], with scores: [0.79, 0.02, 0.19]
# labels: ['product release', 'in-person event', 'company announcement'], with scores: [0.24, 0.11, 0.64]
# labels: ['product release', 'in-person event', 'company announcement'], with scores: [0.09, 0.67, 0.23]

# ToDo
# NLI with NLP for Text-Classification (trained)
# https://huggingface.co/docs/transformers/tasks/sequence_classification
# from transformers import pipeline
# classifier = pipeline("text-classification")
# print(classifier(
#     "Give the release notes of the winter '24 release.",
#     # ToDo (eval, train, infer, etc.)
#     ))