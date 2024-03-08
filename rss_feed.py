import feedparser
import pandas as pd
from datetime import datetime, timedelta

# RSS feed URL
rss_feed_url = "https://feeds.megaphone.fm/spycast"

# Parse the RSS feed
feed = feedparser.parse(rss_feed_url)

# Initialize lists to store data
titles = []
pubDates = []
links = []

# Extract data from the feed
for entry in feed.entries:
    titles.append(entry.title)
    pubDates.append(entry.published)
    link = entry.get('link', None)  # Check if 'link' attribute exists
    links.append(link)

# Convert publication dates to datetime objects
pubDates = [datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z") for date in pubDates]

# Calculate the date 60 days ago from today and make it timezone-aware
cutoff_date = datetime.now().astimezone(pubDates[0].tzinfo) - timedelta(days=60)

# Filter items published in the last 60 days
filtered_titles = [title for title, date in zip(titles, pubDates) if date >= cutoff_date]
filtered_pubDates = [date.strftime("%Y-%m-%d") for date in pubDates if date >= cutoff_date]
# filtered_pubDates = [date.strftime("%a, %d %b %Y %H:%M:%S %z") for date in pubDates if date >= cutoff_date]
filtered_links = [link for link, date in zip(links, pubDates) if date >= cutoff_date]

# Create DataFrame
df_podcast = pd.DataFrame({
    'Title': filtered_titles,
    'PubDate': filtered_pubDates,
    'Link': filtered_links
})


rss_feed_urls = [
    "https://www.economist.com/international/rss.xml",
    "https://www.foreignaffairs.com/rss.xml",
    'https://foreignpolicy.com/feed/'
]

# Initialize lists to store data
titles = []
pubDates = []
links = []

# Process each RSS feed
for rss_feed_url in rss_feed_urls:
    # Parse the RSS feed
    feed = feedparser.parse(rss_feed_url)

    # Extract data from the feed
    for entry in feed.entries:
        titles.append(entry.title)
        pubDates.append(entry.published)
        link = entry.get('link', None)  # Check if 'link' attribute exists
        links.append(link)

# Convert publication dates to datetime objects
pubDates = [datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z") for date in pubDates]

# Calculate the date 60 days ago from today and make it timezone-aware
cutoff_date = datetime.now().astimezone(pubDates[0].tzinfo) - timedelta(days=60)

# Filter items published in the last 60 days
filtered_titles = [title for title, date in zip(titles, pubDates) if date >= cutoff_date]
filtered_pubDates = [date.strftime("%Y-%m-%d") for date in pubDates if date >= cutoff_date]
# filtered_pubDates = [date.strftime("%a, %d %b %Y %H:%M:%S %z") for date in pubDates if date >= cutoff_date]
filtered_links = [link for link, date in zip(links, pubDates) if date >= cutoff_date]

# Create DataFrame
df_magazines = pd.DataFrame({
    'Title': filtered_titles,
    'PubDate': filtered_pubDates,
    'Link': filtered_links 
})

# Keywords to filter
keywords = [
    'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert',
    'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis', 'spying', 'spies', ' cia ', 'mi6',
    "cia'"
]

# Filter DataFrame based on keywords
df_magazines = df_magazines[df_magazines['Title'].str.contains('|'.join(keywords), case=False)]

