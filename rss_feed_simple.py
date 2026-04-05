import feedparser
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Podcasts — Spycast
# ---------------------------------------------------------------------------
rss_feed_url = "https://feeds.megaphone.fm/spycast"
feed = feedparser.parse(rss_feed_url)

titles, pubDates, links = [], [], []
for entry in feed.entries:
    titles.append(entry.title)
    pubDates.append(entry.published)
    links.append(entry.get('link', None))

pubDates = [datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z") for date in pubDates]
cutoff_date = datetime.now().astimezone(pubDates[0].tzinfo) - timedelta(days=60)

df_podcast = pd.DataFrame({
    'Title': [t for t, d in zip(titles, pubDates) if d >= cutoff_date],
    'PubDate': [d.strftime("%Y-%m-%d") for d in pubDates if d >= cutoff_date],
    'Link': [l for l, d in zip(links, pubDates) if d >= cutoff_date],
})

# ---------------------------------------------------------------------------
# Magazines — Economist, Foreign Affairs, Foreign Policy
# ---------------------------------------------------------------------------
rss_feed_urls = [
    "https://www.economist.com/international/rss.xml",
    "https://www.foreignaffairs.com/rss.xml",
    'https://foreignpolicy.com/feed/',
]

titles, pubDates, links = [], [], []
for url in rss_feed_urls:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        titles.append(entry.title)
        pubDates.append(entry.published)
        links.append(entry.get('link', None))

pubDates = [datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z") for date in pubDates]
cutoff_date = datetime.now().astimezone(pubDates[0].tzinfo) - timedelta(days=60)

df_magazines = pd.DataFrame({
    'Title': [t for t, d in zip(titles, pubDates) if d >= cutoff_date],
    'PubDate': [d.strftime("%Y-%m-%d") for d in pubDates if d >= cutoff_date],
    'Link': [l for l, d in zip(links, pubDates) if d >= cutoff_date],
})

keywords = [
    'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert',
    'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis', 'spying',
    'spies', ' cia ', 'mi6', "cia'"
]
df_magazines = df_magazines[df_magazines['Title'].str.contains('|'.join(keywords), case=False)]