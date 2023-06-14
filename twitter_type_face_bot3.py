import csv
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

# Download the stopwords list if not already downloaded
nltk.download('stopwords')

filename = 'Twitter Jan Mar.csv'

# Lists to store the relevant data
tweet_content = []
like_counts = Counter()
retweet_counts = Counter()
most_freq_counts = Counter()

# Create the stopwords list
stopwords_list = set(stopwords.words('english'))
stopwords_list.update(["is", "the", "like", "1/", "\"", "\'\'", "https"])

# Open the CSV file
with open(filename, 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Skip the header row
    next(reader)

    # Process rows in chunks of 1000
    rows_processed = 0
    chunk_size = 100000

    for row in reader:
        # Check if the row has the expected number of columns
        if len(row) >= 6:
            content = row[2]
            like_count = int(row[4])
            retweet_count = int(row[5])

            # Add the content to the tweet_content list
            tweet_content.append(content)

            # Tokenize the content
            tokens = nltk.word_tokenize(content)

            # Remove stopwords and convert to lowercase
            filtered_tokens = [token.lower() for token in tokens if len(token) > 1 and token.lower() not in stopwords_list]

            # Update like_counts and retweet_counts
            if like_count:
                for k in filtered_tokens:
                    if k not in like_counts:
                        like_counts[k] = 0
                    like_counts[k] += like_count
            if retweet_count:
                for k in filtered_tokens:
                    if k not in retweet_counts:
                        retweet_counts[k] = 0
                    retweet_counts[k] += retweet_count
            for k in filtered_tokens:
                if k not in most_freq_counts:
                    most_freq_counts[k] = 0
                most_freq_counts[k] += 1

        rows_processed += 1

        # Print progress every chunk_size rows
        if rows_processed % chunk_size == 0:
            print(f"Processed {rows_processed} rows")

size = 20
most_repeated_words_weight = 10000
most_liked_words_weight = 1000
most_retweeted_words_weight = 5000
# Find the most repeated words
most_repeated_words = [word for word, count in most_freq_counts.most_common()[:most_repeated_words_weight]]

# Find the most liked tweet words
most_liked_words = [word for word, count in like_counts.most_common()[:most_liked_words_weight]]

# Find the most retweeted words
most_retweeted_words = [word for word, count in retweet_counts.most_common()[:most_retweeted_words_weight]]

# Find the intersection of the three factors
common_trending_words = set(most_repeated_words) & set(most_liked_words) & set(most_retweeted_words)

# Print the common trending words
print("Common Trending Words:")
for word in common_trending_words:
    print(word)

# Optional: Filter tweets based on date/time range and perform word frequency analysis

# Create a single figure with subplots for all three charts
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

# Visualize the most repeated words using a bar chart
top_n_repeated = 10
top_repeated_words = most_repeated_words[:top_n_repeated]
top_repeated_counts = [most_freq_counts[word] for word in top_repeated_words]

ax1.bar(range(len(top_repeated_words)), top_repeated_counts, align='center')
ax1.set_xticks(range(len(top_repeated_words)))
ax1.set_xticklabels(top_repeated_words, rotation=90)
ax1.set_xlabel('Words')
ax1.set_ylabel('Frequency')
ax1.set_title('Most Repeated Words')

# Visualize the most liked words using a bar chart
top_n_liked = 10
top_liked_words = most_liked_words[:top_n_liked]
top_liked_counts = [like_counts[word] for word in top_liked_words]

ax2.bar(range(len(top_liked_words)), top_liked_counts, align='center')
ax2.set_xticks(range(len(top_liked_words)))
ax2.set_xticklabels(top_liked_words, rotation=90)
ax2.set_xlabel('Words')
ax2.set_ylabel('Like Count')
ax2.set_title('Most Liked Words')

# Visualize the most retweeted words using a bar chart
top_n_retweeted = 30
top_retweeted_words = most_retweeted_words[:top_n_retweeted]
top_retweeted_counts = [retweet_counts[word] for word in top_retweeted_words]

ax3.bar(range(len(top_retweeted_words)), top_retweeted_counts, align='center')
ax3.set_xticks(range(len(top_retweeted_words)))
ax3.set_xticklabels(top_retweeted_words, rotation=90)
ax3.set_xlabel('Words')
ax3.set_ylabel('Retweet Count')
ax3.set_title('Most Retweeted Words')

plt.tight_layout()
plt.show()
