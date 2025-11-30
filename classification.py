true_positives = 30
false_positives = 5
false_negatives = 10
true_negatives = 55

accuracy = (true_positives + false_positives) / (true_positives + false_positives + false_negatives + true_negatives)

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Genauigkeit: {accuracy:.2f} oder {accuracy * 100:.2f}")
print(f"Präzision: {precision:.2f} oder {precision * 100:.2f}")
print(f"Recall: {recall:.2f} oder {recall * 100:.2f}")
print(f"F1-Score: {f1_score:.2f} oder {f1_score * 100:.2f}")