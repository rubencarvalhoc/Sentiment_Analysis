# Import necessary libraries
import pandas as pd
import tweepy

#Get better view of the dataset
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

# Load twitter api keys and tokens
auth = tweepy.OAuth1UserHandler(
   "YOUR KEYS AND TOKENS HERE", "YOUR KEYS AND TOKENS HERE", "YOUR KEYS AND TOKENS HERE", "YOUR KEYS AND TOKENS HERE"
)

api = tweepy.API(auth)

# Create empty lists
tweets = []
likes = []
date = []

# Get Bill Gates latest 3200 tweets.
gates_tweets = tweepy.Cursor(api.user_timeline, id='BillGates', count=200, tweet_mode="extended").items(3250)

# Iterate through them to add to its respective list
for tweet in gates_tweets:
    tweets.append(tweet.full_text)
    likes.append(tweet.favorite_count)
    date.append(tweet.created_at)

# Create a dataframe to see if it's properly done
df = pd.DataFrame({'tweets': tweets, 'likes': likes, 'date':date})

print(df.shape)
print(df.head(5))

# Create a csv file for analysis
df.to_csv('bill_gates_tweets.csv', index=False)

