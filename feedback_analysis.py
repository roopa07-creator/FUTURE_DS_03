import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/event_feedback.csv")

# 🔥 FIX: normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Columns after fix:", df.columns)

# Sentiment score
df["sentiment_score"] = df["feedback_text"].apply(
    lambda text: TextBlob(text).sentiment.polarity
)

# Sentiment label
def label_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["sentiment_score"].apply(label_sentiment)

print("\nFeedback with Sentiment:")
print(df)

# Rating vs Sentiment plot
plt.figure()
sns.boxplot(x="rating", y="sentiment_score", data=df)
plt.title("Rating vs Sentiment Score")
plt.xlabel("Rating")
plt.ylabel("Sentiment Score")
plt.show()

# Event-wise summary
summary = df.groupby("event_name").agg(
    avg_rating=("rating", "mean"),
    avg_sentiment=("sentiment_score", "mean")
).reset_index()

print("\nEvent-wise Summary:")
print(summary)