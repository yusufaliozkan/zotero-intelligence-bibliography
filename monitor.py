import pandas as pd
import requests
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from rss_feed import df_podcast, df_magazines

# ---------------------------------------------------------------------------
# CONFIG — set these as GitHub Actions secrets
# ---------------------------------------------------------------------------
GMAIL_USER = os.environ["GMAIL_USER"]        # e.g. yourname@gmail.com
GMAIL_PASSWORD = os.environ["GMAIL_PASSWORD"] # Gmail App Password (not your login password)
RECIPIENT = os.environ.get("RECIPIENT_EMAIL", GMAIL_USER)

# ---------------------------------------------------------------------------
# Keywords and journal lists (same as your Streamlit page)
# ---------------------------------------------------------------------------
keywords = [
    'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert',
    'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis', 'spying',
    'spies', 'surveillance', 'targeted killing', 'cyberespionage', ' cia ',
    'rendition', ' mi6 ', ' mi5 ', ' sis ', 'security service', 'central intelligence'
]

journals_with_filtered_items = [
    'The Historical Journal', 'Journal of Policing, Intelligence and Counter Terrorism',
    'Cold War History', 'RUSI Journal', 'Journal of Strategic Studies', 'War in History',
    'International History Review', 'Journal of Contemporary History', 'Middle Eastern Studies',
    'Diplomacy & Statecraft', 'The international journal of intelligence, security, and public affairs',
    'Cryptologia', 'The Journal of Slavic Military Studies', 'International Affairs',
    'Political Science Quarterly', 'Journal of intelligence, conflict and warfare',
    'The Journal of Conflict Studies', 'Journal of Cold War Studies', 'Survival',
    'Security and Defence Quarterly', 'The Journal of Imperial and Commonwealth History',
    'Review of International Studies', 'Diplomatic History', 'Cambridge Review of International Affairs',
    'Public Policy and Administration', 'Armed Forces & Society', 'Studies in Conflict & Terrorism',
    'The English Historical Review', 'World Politics', 'Israel Affairs',
    'Australian Journal of International Affairs', 'Contemporary British History', 'The Historian',
    'The British Journal of Politics and International Relations', 'Terrorism and Political Violence',
    "Mariner's Mirror", 'Small Wars & Insurgencies', 'Journal of Cyber Policy',
    'South Asia:Journal of South Asian Studies', 'International Journal', 'German Law Journal',
    'American Journal of International Law', 'European Journal of International Law',
    'Human Rights Law Review', 'Leiden Journal of International Law',
    'International & Comparative Law Quarterly', 'Journal of Conflict and Security Law',
    'Journal of International Dispute Settlement', 'Security and Human Rights', 'Modern Law Review',
    'International Theory', 'Michigan Journal of International Law', 'Journal of Global Security Studies',
    'Intelligence Studies and Analysis in Modern Context'
]

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

# ---------------------------------------------------------------------------
# Fetch and filter articles from OpenAlex
# ---------------------------------------------------------------------------
def fetch_articles():
    dfs = []
    today = datetime.today().date()

    for api_link in api_links:
        try:
            response = requests.get(api_link, timeout=15)
            if response.status_code != 200:
                continue
            results = response.json().get('results', [])
        except Exception:
            continue

        titles, dois, pub_dates, dois_short, journals = [], [], [], [], []

        for result in results:
            if not result:
                continue
            pub_date_str = result.get('publication_date')
            if not pub_date_str:
                continue
            try:
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
            except ValueError:
                continue
            if today - pub_date > timedelta(days=90):
                continue

            title = result.get('title', '')
            if not title or not any(kw in title.lower() for kw in keywords):
                continue

            primary_location = result.get('primary_location') or {}
            source = primary_location.get('source') or {}
            journal_name = source.get('display_name', 'Unknown')

            doi_full = result.get('doi', 'Unknown')
            doi_short = doi_full.split("https://doi.org/")[-1] if doi_full != 'Unknown' else 'Unknown'

            titles.append(title)
            dois.append(doi_full)
            pub_dates.append(pub_date_str)
            dois_short.append(doi_short)
            journals.append(journal_name)

        if titles:
            dfs.append(pd.DataFrame({
                'Title': titles,
                'Link': dois,
                'Publication Date': pub_dates,
                'DOI': dois_short,
                'Journal': journals,
            }))

    if not dfs:
        return pd.DataFrame()

    final_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Link')

    # Split by journal type
    filtered = final_df[final_df['Journal'].isin(journals_with_filtered_items)]
    other = final_df[~final_df['Journal'].isin(journals_with_filtered_items)]
    return pd.concat([other, filtered], ignore_index=True)


# ---------------------------------------------------------------------------
# Deduplicate against existing library (all_items.csv)
# ---------------------------------------------------------------------------
def deduplicate(filtered_final_df):
    df_dedup = pd.read_csv('all_items.csv')

    # DOI filter
    df_dois = df_dedup.dropna(subset=['DOI'])[['DOI']].copy()
    filtered_final_df['DOI'] = filtered_final_df['DOI'].str.lower()
    df_dois['DOI'] = df_dois['DOI'].str.lower()
    merged = pd.merge(filtered_final_df, df_dois, on='DOI', how='left', indicator=True)
    not_in_library = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)

    # Title filter (catches items with missing/mismatched DOIs)
    df_titles = df_dedup.dropna(subset=['Title'])[['Title']].copy()
    merged2 = pd.merge(not_in_library, df_titles, on='Title', how='left', indicator=True)
    not_in_library = merged2[merged2['_merge'] == 'left_only'].drop('_merge', axis=1)

    not_in_library['Publication Date'] = pd.to_datetime(not_in_library['Publication Date'])
    return not_in_library.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build HTML email body
# ---------------------------------------------------------------------------
def df_to_html_table(df, link_col='Link', title_col='Title'):
    """Render a DataFrame as a clean HTML table with clickable titles."""
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
      <thead>
        <tr style="background:#f2f2f2;">
          <th style="padding:6px 8px; text-align:left;">Title</th>
          <th style="padding:6px 8px; text-align:left;">Journal</th>
          <th style="padding:6px 8px; text-align:left;">Published</th>
        </tr>
      </thead>
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
      <thead>
        <tr style="background:#f2f2f2;">
          <th style="padding:6px 8px; text-align:left;">Title</th>
          <th style="padding:6px 8px; text-align:left;">Published</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_email(future_df, last_30_df, podcast_df, magazine_df):
    today_str = datetime.today().strftime('%d %B %Y')

    def section(heading, table_html, count):
        badge = f'<span style="background:#d32f2f;color:#fff;border-radius:10px;padding:2px 8px;font-size:12px;margin-left:8px;">{count} new</span>' if count else ''
        return f"""
        <h2 style="font-family:Arial,sans-serif; font-size:15px; color:#333; margin-top:24px;">
            {heading}{badge}
        </h2>
        {table_html}"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif; max-width:900px; margin:auto; padding:16px; color:#333;">
    <h1 style="font-size:18px; border-bottom:2px solid #d32f2f; padding-bottom:8px;">
        IntelArchive Daily Monitor &mdash; {today_str}
    </h1>
    <p style="font-size:13px; color:#666;">
        Items detected by OpenAlex that are not yet in the IntelArchive library.
    </p>

    {section("📅 Future / forthcoming publications", df_to_html_table(future_df), len(future_df))}
    {section("📰 Published in the last 30 days", df_to_html_table(last_30_df), len(last_30_df))}
    {section("🎙️ Podcasts", podcast_to_html_table(podcast_df), len(podcast_df))}
    {section("📖 Magazine articles", podcast_to_html_table(magazine_df), len(magazine_df))}

    <p style="font-size:11px; color:#aaa; margin-top:32px;">
        Sent automatically by IntelArchive monitor · {today_str}
    </p>
    </body></html>"""
    return html


# ---------------------------------------------------------------------------
# Send email via Gmail SMTP
# ---------------------------------------------------------------------------
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
    deduped = deduplicate(raw)

    current_date = datetime.now()
    future_df = deduped[deduped['Publication Date'] >= current_date].reset_index(drop=True)
    last_30_df = deduped[
        (deduped['Publication Date'] <= current_date) &
        (deduped['Publication Date'] >= current_date - timedelta(days=30))
    ].reset_index(drop=True)

    # Podcasts & magazines (same logic as Streamlit page)
    df_lib_titles = pd.read_csv('all_items.csv').dropna(subset=['Title'])[['Title']]

    podcast_merged = pd.merge(df_podcast, df_lib_titles, on='Title', how='left', indicator=True)
    new_podcasts = podcast_merged[podcast_merged['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)

    mag_merged = pd.merge(df_magazines, df_lib_titles, on='Title', how='left', indicator=True)
    new_magazines = mag_merged[mag_merged['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)

    total_new = len(future_df) + len(last_30_df) + len(new_podcasts) + len(new_magazines)
    print(f"Found {total_new} new item(s) total.")

    if total_new == 0:
        print("Nothing new today. No email sent.")
    else:
        html = build_email(future_df, last_30_df, new_podcasts, new_magazines)
        send_email(html, total_new)