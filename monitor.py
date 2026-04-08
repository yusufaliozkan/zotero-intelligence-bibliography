import pandas as pd
import requests
import smtplib
import os
import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from rss_feed_simple import df_podcast, df_magazines

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASSWORD = os.environ["GMAIL_PASSWORD"]
RECIPIENT = os.environ.get("RECIPIENT_EMAIL", GMAIL_USER)

# ---------------------------------------------------------------------------
# Same API links, journals, keywords as the Streamlit page (document 20)
# ---------------------------------------------------------------------------
api_links = [
    'https://api.openalex.org/works?filter=primary_location.source.id:s33269604&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s205284143&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210168073&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s2764506647&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s2764781490&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s93928036&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s962698607&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s199078552&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s145781505&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s120387555&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s161550498&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s164505828&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s99133842&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210219209&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s185196701&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s157188123&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s79519963&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s161027966&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210201145&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s2764954702&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s200077084&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s27717133&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210214688&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s112911512&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s131264395&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s154084123&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s103350616&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s17185278&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s21016770&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s41746314&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s56601287&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s143110675&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s106532728&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s67329160&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s49917718&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s8593340&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s161552967&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s141724154&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s53578506&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210184262&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210236978&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s120889147&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?filter=primary_location.source.id:s86954274&sort=publication_year:desc&per_page=10',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s117224066&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s160097506&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s175405714&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s84944781&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s154337186&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s156235965&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s68909633&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s42104779&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764513295&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s82119083&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s129176075&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764608241&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2735957470&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=primary_topic.id:t12572&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=concepts.id:c558872910&sort=publication_year:desc',
    'https://api.openalex.org/works?page=1&filter=concepts.id:c173127888&sort=publication_year:desc',
]

journals_with_filtered_items = [
    'The Historical Journal', 'Journal of Policing, Intelligence and Counter Terrorism', 'Cold War History', 'RUSI Journal',
    'Journal of Strategic Studies', 'War in History', 'International History Review', 'Journal of Contemporary History',
    'Middle Eastern Studies', 'Diplomacy & Statecraft', 'The international journal of intelligence, security, and public affairs',
    'Cryptologia', 'The Journal of Slavic Military Studies', 'International Affairs', 'Political Science Quarterly',
    'Journal of intelligence, conflict and warfare', 'The Journal of Conflict Studies', 'Journal of Cold War Studies', 'Survival',
    'Security and Defence Quarterly', 'The Journal of Imperial and Commonwealth History', 'Review of International Studies', 'Diplomatic History',
    'Cambridge Review of International Affairs', 'Public Policy and Administration', 'Armed Forces & Society', 'Studies in Conflict & Terrorism',
    'The English Historical Review', 'World Politics', 'Israel Affairs', 'Australian Journal of International Affairs', 'Contemporary British History',
    'The Historian', 'The British Journal of Politics and International Relations', 'Terrorism and Political Violence', "Mariner's Mirror",
    'Small Wars & Insurgencies', 'Journal of Cyber Policy', 'South Asia:Journal of South Asian Studies', 'International Journal', 'German Law Journal',
    'American Journal of International Law', 'European Journal of International Law', 'Human Rights Law Review', 'Leiden Journal of International Law',
    'International & Comparative Law Quarterly', 'Journal of Conflict and Security Law', 'Journal of International Dispute Settlement', 'Security and Human Rights',
    'Modern Law Review', 'International Theory', 'Michigan Journal of International Law', 'Journal of Global Security Studies',
    'Intelligence Studies and Analysis in Modern Context'
]

keywords = [
    'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert', 'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis',
    'spying', 'spies', 'surveillance', 'targeted killing', 'cyberespionage', ' cia ', 'rendition', ' mi6 ', ' mi5 ', ' sis ',
    'security service', 'central intelligence'
]


# ---------------------------------------------------------------------------
# Step 1: Fetch from OpenAlex (identical logic to document 20)
# ---------------------------------------------------------------------------
def fetch_articles():
    dfs = []
    today = datetime.today().date()

    for api_link in api_links:
        response = requests.get(api_link)
        if response.status_code != 200:
            continue

        results = response.json().get('results', [])
        titles, dois, publication_dates, dois_without_https, journals = [], [], [], [], []

        for result in results:
            if result is None:
                continue
            pub_date_str = result.get('publication_date')
            if pub_date_str is None:
                continue
            try:
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
            except ValueError:
                continue
            if today - pub_date <= timedelta(days=90):
                title = result.get('title')
                if title is not None and any(keyword in title.lower() for keyword in keywords):
                    titles.append(title)
                    dois.append(result.get('doi', 'Unknown'))
                    publication_dates.append(pub_date_str)
                    ids = result.get('ids', {})
                    doi_value = ids.get('doi', 'Unknown')
                    dois_without_https.append(doi_value.split("https://doi.org/")[-1] if doi_value != 'Unknown' else 'Unknown')
                    primary_location = result.get('primary_location', {})
                    source = primary_location.get('source')
                    journals.append(source.get('display_name', 'Unknown') if source else 'Unknown')

        if titles:
            dfs.append(pd.DataFrame({
                'Title': titles, 'Link': dois,
                'Publication Date': publication_dates,
                'DOI': dois_without_https, 'Journal': journals,
            }))

    if not dfs:
        return pd.DataFrame()

    final_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Link')
    historical_journal_filtered = final_df[final_df['Journal'].isin(journals_with_filtered_items)]
    other_journals = final_df[~final_df['Journal'].isin(journals_with_filtered_items)]
    return pd.concat([other_journals, historical_journal_filtered], ignore_index=True)


# ---------------------------------------------------------------------------
# Step 2: Deduplicate against all_items.csv (identical logic to document 20)
# ---------------------------------------------------------------------------
def deduplicate(filtered_final_df):
    df_dedup = pd.read_csv('all_items.csv')

    # DOI filter
    df_dois = df_dedup.copy()
    df_dois.dropna(subset=['DOI'], inplace=True)
    df_dois = df_dois[['DOI']].reset_index(drop=True)
    filtered_final_df['DOI'] = filtered_final_df['DOI'].str.lower()
    df_dois['DOI'] = df_dois['DOI'].str.lower()
    merged_df = pd.merge(filtered_final_df, df_dois[['DOI']], on='DOI', how='left', indicator=True)
    items_not_in_df2 = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
    words_to_exclude = ['notwantedwordshere']
    mask = ~items_not_in_df2['Title'].str.contains('|'.join(words_to_exclude), case=False)
    items_not_in_df2 = items_not_in_df2[mask].reset_index(drop=True)

    # Title filter
    df_titles = df_dedup.copy()
    df_titles.dropna(subset=['Title'], inplace=True)
    df_titles = df_titles[['Title']].reset_index(drop=True)
    merged_df_2 = pd.merge(items_not_in_df2, df_titles[['Title']], on='Title', how='left', indicator=True)
    items_not_in_df3 = merged_df_2[merged_df_2['_merge'] == 'left_only'].drop('_merge', axis=1)
    items_not_in_df3 = items_not_in_df3.sort_values(by=['Publication Date'], ascending=False).reset_index(drop=True)

    return items_not_in_df3, df_titles


# ---------------------------------------------------------------------------
# Step 3: Podcasts & magazines (identical logic to document 20)
# ---------------------------------------------------------------------------
def get_podcasts_magazines(dismissed_titles=set()):
    df_dedup = pd.read_csv('all_items.csv')
    df_lib_titles = df_dedup.dropna(subset=['Title'])[['Title']].copy()

    podcast_merged = pd.merge(df_podcast, df_lib_titles, on='Title', how='left', indicator=True)
    new_podcasts = podcast_merged[podcast_merged['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)
    if dismissed_titles:
        new_podcasts = new_podcasts[~new_podcasts['Title'].str.lower().isin(dismissed_titles)].reset_index(drop=True)

    mag_merged = pd.merge(df_magazines, df_lib_titles, on='Title', how='left', indicator=True)
    new_magazines = mag_merged[mag_merged['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)
    if dismissed_titles:
        new_magazines = new_magazines[~new_magazines['Title'].str.lower().isin(dismissed_titles)].reset_index(drop=True)

    return new_podcasts, new_magazines


# ---------------------------------------------------------------------------
# Step 4: Other resources / RSS (identical logic to document 20)
# ---------------------------------------------------------------------------
def get_other_resources(df_titles, dismissed_titles=set()):
    feeds = [{"url": "https://www.aspistrategist.org.au/feed/", "label": "Australian Strategic Policy Institute"}]
    all_data = []
    for feed in feeds:
        try:
            response = requests.get(feed["url"], timeout=15)
            root = ET.fromstring(response.content)
            for item in root.findall('.//item')[1:]:
                all_data.append({
                    'title': item.find('title').text,
                    'link': item.find('link').text,
                    'label': feed["label"],
                    'pubDate': item.find('pubDate').text
                })
        except Exception as e:
            print(f"RSS fetch error: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    words_to_filter = ["intelligence", "espionage", "spy", "oversight"]
    df = df[df['title'].str.contains('|'.join(words_to_filter), case=False, na=False)].reset_index(drop=True)
    df = df.rename(columns={'title': 'Title'})
    df['Title'] = df['Title'].str.upper()
    df_titles_upper = df_titles.copy()
    df_titles_upper['Title'] = df_titles_upper['Title'].str.upper()

    def find_similar_title(title, titles, threshold=80):
        for t in titles:
            if fuzz.ratio(title, t) >= threshold:
                return t
        return None

    df['Similar_Title'] = df['Title'].apply(lambda x: find_similar_title(x, df_titles_upper['Title'], threshold=80))
    df_not = df.merge(df_titles_upper[['Title']], left_on='Similar_Title', right_on='Title', how='left', indicator=True)
    df_not = df_not[df_not['_merge'] == 'left_only'].drop(['_merge', 'Similar_Title'], axis=1).reset_index(drop=True)

    if 'Title_x' in df_not.columns:
        df_not = df_not.drop('Title_y', axis=1).rename(columns={'Title_x': 'Title'})

    if dismissed_titles:
        df_not = df_not[~df_not['Title'].str.lower().isin(dismissed_titles)].reset_index(drop=True)

    return df_not


# ---------------------------------------------------------------------------
# Email functions
# ---------------------------------------------------------------------------
def df_to_html_table(df, link_col='Link', title_col='Title'):
    if df.empty:
        return '<p style="color:#888;">No items found.</p>'
    rows = ''
    for _, row in df.iterrows():
        link = row.get(link_col, '#')
        title = row.get(title_col, 'Untitled')
        journal = row.get('Journal', '')
        pub_date = str(row.get('Publication Date', ''))[:10]
        rows += f"""
        <tr>
          <td style="padding:6px 8px; border-bottom:1px solid #eee;">
            <a href="{link}" style="color:#1a0dab; text-decoration:none;">{title}</a>
          </td>
          <td style="padding:6px 8px; border-bottom:1px solid #eee; color:#555; white-space:nowrap;">{journal}</td>
          <td style="padding:6px 8px; border-bottom:1px solid #eee; color:#555; white-space:nowrap;">{pub_date}</td>
        </tr>"""
    return f"""
    <table style="border-collapse:collapse; width:100%; font-family:Arial,sans-serif; font-size:13px;">
      <thead><tr style="background:#f2f2f2;">
        <th style="padding:6px 8px; text-align:left;">Title</th>
        <th style="padding:6px 8px; text-align:left;">Journal</th>
        <th style="padding:6px 8px; text-align:left;">Published</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def podcast_to_html_table(df):
    if df.empty:
        return '<p style="color:#888;">No new items found.</p>'
    rows = ''
    for _, row in df.iterrows():
        title = row.get('Title', 'Untitled')
        link = row.get('Link', '#')
        pub_date = str(row.get('PubDate', ''))[:10]
        rows += f"""
        <tr>
          <td style="padding:6px 8px; border-bottom:1px solid #eee;">
            <a href="{link}" style="color:#1a0dab; text-decoration:none;">{title}</a>
          </td>
          <td style="padding:6px 8px; border-bottom:1px solid #eee; color:#555; white-space:nowrap;">{pub_date}</td>
        </tr>"""
    return f"""
    <table style="border-collapse:collapse; width:100%; font-family:Arial,sans-serif; font-size:13px;">
      <thead><tr style="background:#f2f2f2;">
        <th style="padding:6px 8px; text-align:left;">Title</th>
        <th style="padding:6px 8px; text-align:left;">Published</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_email(future_df, last_30_df, last_90_df, podcast_df, magazine_df):
    today_str = datetime.today().strftime('%d %B %Y')

    def section(heading, table_html, count):
        badge = f'<span style="background:#d32f2f;color:#fff;border-radius:10px;padding:2px 8px;font-size:12px;margin-left:8px;">{count} new</span>' if count else ''
        return f'<h2 style="font-family:Arial,sans-serif;font-size:15px;color:#333;margin-top:24px;">{heading}{badge}</h2>{table_html}'

    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:900px;margin:auto;padding:16px;color:#333;">
    <h1 style="font-size:18px;border-bottom:2px solid #d32f2f;padding-bottom:8px;">
        IntelArchive Daily Monitor &mdash; {today_str}
    </h1>
    <p style="font-size:13px;color:#666;">Items detected by OpenAlex not yet in the IntelArchive library.</p>
    {section("📅 Future / forthcoming publications", df_to_html_table(future_df), len(future_df))}
    {section("📰 Published in the last 30 days", df_to_html_table(last_30_df), len(last_30_df))}
    {section("🗂️ Published in the last 31–90 days", df_to_html_table(last_90_df), len(last_90_df))}
    {section("🎙️ Podcasts", podcast_to_html_table(podcast_df), len(podcast_df))}
    {section("📖 Magazine articles", podcast_to_html_table(magazine_df), len(magazine_df))}
    <p style="font-size:11px;color:#aaa;margin-top:32px;">Sent automatically by IntelArchive monitor · {today_str}</p>
    </body></html>"""


def send_email(html_body, total_new):
    today_str = datetime.today().strftime('%d %B %Y')
    subject = f"IntelArchive Monitor — {total_new} new item(s) — {today_str}"
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = GMAIL_USER
    msg['To'] = RECIPIENT
    msg.attach(MIMEText(html_body, 'html'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.sendmail(GMAIL_USER, RECIPIENT, msg.as_string())
    print(f"Email sent: {subject}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Fetching articles from OpenAlex...")
    raw = fetch_articles()

    if raw.empty:
        print("No articles fetched. Exiting.")
        exit(0)

    print(f"Fetched {len(raw)} candidate articles. Deduplicating...")
    deduped, df_titles = deduplicate(raw)
    print(f"After deduplication: {len(deduped)} articles remaining.")

    # Filter out dismissed items
    dismissed_titles = set()
    dismissed_dois = set()
    try:
        dismissed = pd.read_csv('dismissed.csv')
        if not dismissed.empty:
            dismissed_dois = set(dismissed['DOI'].str.lower().dropna())
            dismissed_titles = set(dismissed['Title'].str.lower().dropna())
            deduped = deduped[
                ~deduped['DOI'].str.lower().isin(dismissed_dois) &
                ~deduped['Title'].str.lower().isin(dismissed_titles)
            ].reset_index(drop=True)
            print(f"After dismissed filter: {len(deduped)} articles remaining.")
    except FileNotFoundError:
        print("No dismissed.csv found, skipping.")

    # Split into future / last 30 days (identical to document 20)
    deduped['Publication Date'] = pd.to_datetime(deduped['Publication Date'])
    current_date = datetime.now()
    future_df = deduped[deduped['Publication Date'] >= current_date].reset_index(drop=True)
    last_30_df = deduped[
        (deduped['Publication Date'] <= current_date) &
        (deduped['Publication Date'] >= current_date - timedelta(days=30))
    ].reset_index(drop=True)

    last_90_df = deduped[
        (deduped['Publication Date'] <= current_date - timedelta(days=30)) &
        (deduped['Publication Date'] >= current_date - timedelta(days=90))
    ].reset_index(drop=True)

    # Podcasts & magazines
    new_podcasts, new_magazines = get_podcasts_magazines(dismissed_titles)

    # Other resources
    df_not = get_other_resources(df_titles, dismissed_titles)

    # ---------------------------------------------------------------------------
    # Save all results to CSV files for the Streamlit page to read
    # ---------------------------------------------------------------------------
    generated_at = datetime.now().isoformat()
    future_df['generated_at'] = generated_at
    last_30_df['generated_at'] = generated_at
    last_90_df['generated_at'] = generated_at

    future_df.to_csv('monitor_future.csv', index=False)
    last_30_df.to_csv('monitor_last30.csv', index=False)
    new_podcasts.to_csv('monitor_podcasts.csv', index=False)
    new_magazines.to_csv('monitor_magazines.csv', index=False)
    df_not.to_csv('monitor_other.csv', index=False)
    last_90_df.to_csv('monitor_last90.csv', index=False)

    print(f"Future: {len(future_df)}, Last 30: {len(last_30_df)}, Last 90: {len(last_90_df)}, Podcasts: {len(new_podcasts)}, Magazines: {len(new_magazines)}, Other: {len(df_not)}")

    # Send email
    total_new = len(future_df) + len(last_30_df) + len(last_90_df) + len(new_podcasts) + len(new_magazines)
    print(f"Found {total_new} new item(s) total.")

    if total_new == 0:
        print("Nothing new today. No email sent.")
    else:
        html = build_email(future_df, last_30_df, last_90_df, new_podcasts, new_magazines)
        send_email(html, total_new)