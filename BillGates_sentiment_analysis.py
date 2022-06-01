# Import necessary libraries
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import emoji
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

# Get a better view of the dataset
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

# Load data
df = pd.read_csv(r"C:\Users\PC\Desktop\PycharmProjects\DataAnalysis\bill_gates_tweets.csv")

# View our data
print(df.head(5))
print(df.dtypes)

# Let's parse our 'date' column
df['date_parsed'] = pd.to_datetime(df['date'], infer_datetime_format=True)

# Now we can create a column that has the hour (UTC±00:00) that the tweet was created
df['publish_hour'] = df['date_parsed'].dt.hour

# We can drop the 'date' object column now
df.drop(['date'], axis=1, inplace=True)

# Let's create our function to clean the tweets
def tweet_cleaner(words):
    # This will delete any emojis from the sentences
    words = emoji.replace_emoji(words, replace="")
    # This will transform all words to lowercase
    words = words.lower()
    # This will delete all mentions and hashtags
    words = re.sub("@[A-Za-z0-9]+", "", words)
    words = re.sub("#[A-Za-z0-9_]+", "", words)
    # This will delete links
    words = re.sub(r"http\S+", "", words)
    words = re.sub(r"www.\S+", "", words)
    # Delete character which are not a word character
    words = re.sub('\W+', ' ', words)
    return words

# Create new column with preprocessed text
df['tweets_cleaned'] = df['tweets'].map(lambda x: tweet_cleaner(x))
# Drop null values
df.dropna(inplace=True)

print(df.head(5))

# Create celaned tweets list
tweets_cleaned_list = []
for clean in df['tweets_cleaned']:
    tweets_cleaned_list.append(clean)

print("Raw tweets: \n", df['tweets'][0])
print("Cleaned tweets:\n", df['tweets_cleaned'][0])

# Text preprocessing
sentences_tokenized = [word_tokenize(tokens) for tokens in tweets_cleaned_list]
print("Token tweets:\n", sentences_tokenized[0])

# Different tokenization, this one removes stopwords
sentences_tokenized2 = []
for words in sentences_tokenized:
    w = []
    for word in words:
        if not word in stopwords.words('english'):
            w.append(word)
    sentences_tokenized2.append(w)

print("Removed stopwords:\n", sentences_tokenized2[0])

# Stemming the words
stemmer = PorterStemmer()
stemmed = []
for words in sentences_tokenized:
    w = []
    for word in words:
        w.append(stemmer.stem(word))
    stemmed.append(w)

print("Stemmization:\n", stemmed[0])

# Lemmatization of words
lemmatizer = WordNetLemmatizer()
lemmatized = []
for i in sentences_tokenized:
    w = []
    for x in i:
        w.append(lemmatizer.lemmatize(x))
    lemmatized.append(w)

print("Lemmatization: \n", lemmatized[0])

print(df.head(5))

lem=[]
for i in lemmatized:
    x = ' '.join(i)
    lem.append(x)

df['tweets_lem'] = lem
print(df.head(5))
print(df.shape)
analyzer = SentimentIntensityAnalyzer()
# Create a compound column with Vader
df['compound'] = df['tweets_lem'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
# Increase readability
df['sentiment'] = df['compound'].map(lambda x: "neutral" if x > 0.05 and x < 0.05 else("positive" if x >= 0.05 else "negative"))

# Get the number of words per tweet
df['number_of_words'] = df['tweets_cleaned'].apply(lambda x: len(str(x).split(" ")))

print(df.head(5))

# Type of tweet percentage the dataframe is composed of
positive_tweets = df.loc[df.compound >= 0.05]
percentage_positive = (len(positive_tweets) / len(df.compound)) * 100

negative_tweets = df.loc[df.compound <= 0.05]
percentage_negative = (len(negative_tweets) / len(df.compound)) * 100

neutral_tweets = df.loc[(df.compound > 0.05) & (df.compound < 0.05)]
percentage_neutral = (len(neutral_tweets) / len(df.compound)) * 100

print(f"Percentage of positive tweets: {round(percentage_positive, 2)}")
print(f"Percentage of negative tweets: {round(percentage_negative, 2)}")
print(f"Percentage of netural tweets: {round(percentage_neutral, 2)}")


# Let's create some graphs!
plt.figure(figsize=(10,8))
sns.histplot(df['number_of_words'])
plt.title("Sentence length by words", fontsize=15)
plt.ylabel("Number of tweets", fontsize=12)
plt.xlabel("Number of words", fontsize=12)
#plt.show()

plt.figure(figsize=(10,8))
sns.countplot(data=df, x='sentiment')
plt.title("Tweet sentiment", fontsize=15)
plt.ylabel("Number of tweets", fontsize=12)
plt.xlabel("Sentiment", fontsize=12)
#plt.show()

plt.figure(figsize=(10,8))
sns.barplot(data=df, x='publish_hour', y='likes', palette='crest')
plt.title("Tweet likes per hour (UTC±00:00)", fontsize=15)
plt.ylabel("Number of likes", fontsize=12)
plt.xlabel("Hour that tweet was published", fontsize=12)
#plt.show()

plt.figure(figsize=(10,8))
sns.set_theme(style='darkgrid')
sns.lineplot(data=df, x='publish_hour', y='likes')
plt.title("Tweet likes per hour (UTC±00:00)", fontsize=15)
plt.ylabel("Number of likes", fontsize=12)
plt.xlabel("Hour that tweet was published", fontsize=12)
#plt.show()


# Let's build our model that tweet sentiment
x = df['tweets_cleaned']
y = df['likes']

# CountVectorizer to create our BoW
cv = CountVectorizer(max_features=1500, analyzer='word', ngram_range=(1, 3))


# Simple model
x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=42)

x_train_cv = cv.fit_transform(x_train)
x_valid_cv = cv.transform(x_valid)

from sklearn import svm

svr = svm.SVR()
svr.fit(x_train_cv, y_train)

from sklearn.model_selection import cross_val_score
def get_score(model):
    return -1 * cross_val_score(model, x_train_cv, y_train, cv=10, scoring='neg_mean_absolute_error').mean()

print(get_score(svr))

