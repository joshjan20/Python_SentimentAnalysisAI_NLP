# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sentiment_analysis as analyser

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Step 2: Data Preparation
    analyser.download_data()
    df = analyser.load_data()

    # Step 3: Model Training
    model, vectorizer = analyser.train_model(df)

    # Step 4: Prediction
    test_reviews = [
        "I absolutely loved this movie! It was fantastic.",
        "It was a terrible film. I hated it.",
        "The movie was okay, nothing special."
    ]

    for review in test_reviews:
        sentiment = analyser.predict_sentiment(model, vectorizer, review)
        print(f"Review: '{review}' | Sentiment: {sentiment}")
