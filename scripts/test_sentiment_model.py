import json

from sentiment.inference import predict_sentiment

samples = [
    "Pelayanannya sangat ramah dan makanannya enak sekali.",
    "The service was wonderful and the food was excellent.",
]

for text in samples:
    print("\nTeks:", text)
    result = predict_sentiment(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
