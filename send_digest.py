import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import os

BASE_URL = "https://intelligence.streamlit.app"
EVENTS_URL = f"{BASE_URL}/Events"

SHEET_ID = "10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI"

def sheet_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

GID_EVENTS_MAIN  = "0"
GID_EVENTS_FORMS = "1941981997"
GID_CONF_MAIN    = "939232836"
GID_CONF_V2      = "312814443"
GID_CFP_MAIN     = "135096406"
GID_CFP_V2       = "1589739166"

def get_new_items(days=7):
    df = pd.read_csv("all_items.csv")
    df["Date added"] = pd.to_datetime(df["Date added"], utc=True, errors="coerce")
    cutoff = datetime.now(tz=df["Date added"].dt.tz) - timedelta(days=days)
    new_items = df[df["Date added"] >= cutoff].copy()
    new_items = new_items.sort_values("Date added", ascending=False)
    return new_items

def get_upcoming_events():
    today = pd.Timestamp.now().normalize()
    try:
        df1 = pd.read_csv(sheet_url(GID_EVENTS_MAIN))
        df2 = pd.read_csv(sheet_url(GID_EVENTS_FORMS))
        df2 = df2.rename(columns={
            'Event name': 'event_name',
            'Event organiser': 'organiser',
            'Link to the event': 'link',
            'Date of event': 'date',
            'Event venue': 'venue',
            'Details': 'details'
        })
        df = pd.concat([df1, df2], axis=0)
        df = df.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        df = df[
            (df['date'] >= today) &
            (df['date'] <= today + pd.Timedelta(days=30))
        ].sort_values('date')
        print(f"Events after parse: {len(df)} rows")
        df = df[df['date'] >= today].sort_values('date')
        print(f"Events after filter: {len(df)} rows")
        df['date_display'] = df['date'].dt.strftime('%d %B %Y')
        df['venue']     = df['venue'].fillna('').astype(str)
        df['organiser'] = df['organiser'].fillna('').astype(str)
        df['link']      = df['link'].fillna('').astype(str)
        return df
    except Exception as e:
        print(f"Error fetching events: {e}")
        return pd.DataFrame()

def get_upcoming_conferences():
    today = pd.Timestamp.now().normalize()
    try:
        df1 = pd.read_csv(sheet_url(GID_CONF_MAIN))
        df2 = pd.read_csv(sheet_url(GID_CONF_V2))
        if 'Timestamp' in df2.columns:
            df2 = df2.drop('Timestamp', axis=1)
        df = pd.concat([df1, df2], axis=0)
        df = df.drop_duplicates(subset=['link'], keep='first')
        df['date']     = pd.to_datetime(df['date'],     format='%m/%d/%Y', errors='coerce')
        df['date_end'] = pd.to_datetime(df['date_end'], format='%m/%d/%Y', errors='coerce')
        df['date_end'] = df['date_end'].fillna(df['date'])
        df = df[
            (df['date_end'] >= today) &
            (df['date'] <= today + pd.Timedelta(days=90))
        ].sort_values('date')
        print(f"Conferences after parse: {len(df)} rows")
        print(df[['conference_name', 'date', 'date_end']].to_string())
        df = df[df['date_end'] >= today].sort_values('date')
        print(f"Conferences after filter: {len(df)} rows")
        df['date_display']     = df['date'].dt.strftime('%d %B %Y')
        df['date_end_display'] = df['date_end'].dt.strftime('%d %B %Y')
        df['venue']     = df['venue'].fillna('').astype(str)
        df['organiser'] = df['organiser'].fillna('').astype(str)
        df['link']      = df['link'].fillna('').astype(str)
        return df
    except Exception as e:
        print(f"Error fetching conferences: {e}")
        return pd.DataFrame()

def get_upcoming_cfp():
    today = pd.Timestamp.now().normalize()
    try:
        df1 = pd.read_csv(sheet_url(GID_CFP_MAIN))
        df2 = pd.read_csv(sheet_url(GID_CFP_V2))
        df = pd.concat([df1, df2], axis=0)
        df = df.drop_duplicates(subset=['name', 'link', 'deadline'], keep='first')
        df['deadline'] = pd.to_datetime(df['deadline'], format='%m/%d/%Y', errors='coerce')
        print(f"CFPs after parse: {len(df)} rows")
        df = df[df['deadline'] >= today].sort_values('deadline')
        print(f"CFPs after filter: {len(df)} rows")
        df['deadline_display'] = df['deadline'].dt.strftime('%d %B %Y')
        df['organiser'] = df['organiser'].fillna('').astype(str)
        df['link']      = df['link'].fillna('').astype(str)
        return df
    except Exception as e:
        print(f"Error fetching CFPs: {e}")
        return pd.DataFrame()

def build_events_html(events_df, conferences_df, cfp_df):
    html = ""

    # ── Events ────────────────────────────────────────────────────────────────
    if not events_df.empty:
        html += """
        <h3 style="color: #1a1a1a; border-bottom: 2px solid #5cb85c; padding-bottom: 6px;
                   margin-top: 28px; font-family: Georgia, serif;">
            Upcoming Events (in the next 30 days)
        </h3>
        """
        for _, row in events_df.head(10).iterrows():
            name      = str(row.get('event_name', '')).strip()
            link      = str(row.get('link', '')).strip()
            organiser = str(row.get('organiser', '')).strip()
            date      = str(row.get('date_display', '')).strip()
            venue     = str(row.get('venue', '')).strip()
            title_html = f'<a href="{link}" style="font-weight: bold; color: #1a1a1a; text-decoration: none; font-family: Georgia, serif; font-size: 1em;">{name}</a>' if link else f'<strong>{name}</strong>'
            html += f"""
            <div style="margin-bottom: 12px; padding: 12px 16px; background: #f8f8f8;
                        border-left: 4px solid #5cb85c; border-radius: 0 4px 4px 0;">
                {title_html}<br>
                <span style="color: #555; font-size: 0.88em; font-family: Arial, sans-serif;">
                    {organiser} &nbsp;·&nbsp; {date}
                    {"&nbsp;·&nbsp;" + venue if venue else ""}
                </span>
            </div>
            """
        if len(events_df) > 10:
            html += f'<p style="font-family: Arial, sans-serif; font-size: 0.85em; color: #888;">And {len(events_df) - 10} more events. <a href="{EVENTS_URL}" style="color: #5cb85c;">See all →</a></p>'

    # ── Conferences ───────────────────────────────────────────────────────────
    if not conferences_df.empty:
        html += """
        <h3 style="color: #1a1a1a; border-bottom: 2px solid #5cb85c; padding-bottom: 6px;
                   margin-top: 28px; font-family: Georgia, serif;">
            Upcoming Conferences (in the next 90 days)
        </h3>
        """
        for _, row in conferences_df.head(10).iterrows():
            name      = str(row.get('conference_name', '')).strip()
            link      = str(row.get('link', '')).strip()
            organiser = str(row.get('organiser', '')).strip()
            date      = str(row.get('date_display', '')).strip()
            date_end  = str(row.get('date_end_display', '')).strip()
            venue     = str(row.get('venue', '')).strip()
            title_html = f'<a href="{link}" style="font-weight: bold; color: #1a1a1a; text-decoration: none; font-family: Georgia, serif; font-size: 1em;">{name}</a>' if link else f'<strong>{name}</strong>'
            html += f"""
            <div style="margin-bottom: 12px; padding: 12px 16px; background: #f8f8f8;
                        border-left: 4px solid #5cb85c; border-radius: 0 4px 4px 0;">
                {title_html}<br>
                <span style="color: #555; font-size: 0.88em; font-family: Arial, sans-serif;">
                    {organiser} &nbsp;·&nbsp; {date} – {date_end}
                    {"&nbsp;·&nbsp;" + venue if venue else ""}
                </span>
            </div>
            """
        if len(conferences_df) > 10:
            html += f'<p style="font-family: Arial, sans-serif; font-size: 0.85em; color: #888;">And {len(conferences_df) - 10} more conferences. <a href="{EVENTS_URL}" style="color: #5cb85c;">See all →</a></p>'

    # ── Call for Papers ───────────────────────────────────────────────────────
    if not cfp_df.empty:
        html += """
        <h3 style="color: #1a1a1a; border-bottom: 2px solid #5cb85c; padding-bottom: 6px;
                   margin-top: 28px; font-family: Georgia, serif;">
            Calls for Papers
        </h3>
        """
        for _, row in cfp_df.head(10).iterrows():
            name      = str(row.get('name', '')).strip()
            link      = str(row.get('link', '')).strip()
            organiser = str(row.get('organiser', '')).strip()
            deadline  = str(row.get('deadline_display', '')).strip()
            title_html = f'<a href="{link}" style="font-weight: bold; color: #1a1a1a; text-decoration: none; font-family: Georgia, serif; font-size: 1em;">{name}</a>' if link else f'<strong>{name}</strong>'
            html += f"""
            <div style="margin-bottom: 12px; padding: 12px 16px; background: #f8f8f8;
                        border-left: 4px solid #5cb85c; border-radius: 0 4px 4px 0;">
                {title_html}<br>
                <span style="color: #555; font-size: 0.88em; font-family: Arial, sans-serif;">
                    {organiser} &nbsp;·&nbsp; Deadline: {deadline}
                </span>
            </div>
            """
        if len(cfp_df) > 10:
            html += f'<p style="font-family: Arial, sans-serif; font-size: 0.85em; color: #888;">And {len(cfp_df) - 10} more calls. <a href="{EVENTS_URL}" style="color: #5cb85c;">See all →</a></p>'

    return html

def build_html_digest(df):
    today = datetime.now().strftime("%d %B %Y")
    count = len(df)

    if count == 0:
        return None

    grouped = df.groupby("Publication type")

    rows_html = ""
    for pub_type, group in grouped:
        rows_html += f"""
        <h3 style="color: #1a1a1a; border-bottom: 2px solid #5cb85c; padding-bottom: 6px; margin-top: 28px; font-family: Georgia, serif;">
            {pub_type} <span style="color: #888; font-size: 0.85em;">({len(group)})</span>
        </h3>
        """
        for _, row in group.iterrows():
            title     = str(row.get("Title", "")).strip()
            authors   = str(row.get("FirstName2", "")).strip()
            date_pub  = str(row.get("Date published", "")).strip()
            journal   = str(row.get("Journal", "")).strip()
            publisher = str(row.get("Publisher", "")).strip()
            zotero    = str(row.get("Zotero link", "")).strip()

            parent_key = zotero.rstrip("/").split("/")[-1] if zotero else ""
            item_url   = f"{BASE_URL}/?item={parent_key}" if parent_key else BASE_URL

            if journal and journal != "nan":
                source = f"<em>{journal}</em>"
            elif publisher and publisher != "nan":
                source = f"{publisher}"
            else:
                source = ""

            authors_display = authors if authors and authors != "nan" else "N/A"
            if date_pub and date_pub != "nan":
                try:
                    date_display = pd.to_datetime(date_pub).strftime("%d %B %Y")
                except Exception:
                    date_display = date_pub
            else:
                date_display = "N/A"

            link_to_publication = str(row.get("Link to publication", "")).strip()
            if link_to_publication == "nan":
                link_to_publication = ""

            rows_html += f"""
            <div style="margin-bottom: 14px; padding: 14px 16px; background: #f8f8f8; border-left: 4px solid #5cb85c; border-radius: 0 4px 4px 0;">
                <a href="{item_url}" style="font-weight: bold; color: #1a1a1a; text-decoration: none; font-family: Georgia, serif; font-size: 1em; line-height: 1.4;">
                    {title}
                </a><br>
                <span style="color: #555; font-size: 0.88em; font-family: Arial, sans-serif;">
                    {authors_display} &nbsp;·&nbsp; {date_display}
                    {"&nbsp;·&nbsp;" + source if source else ""}
                </span><br>
                <a href="{item_url}" style="font-size: 0.82em; color: #5cb85c; text-decoration: none; font-family: Arial, sans-serif;">
                    View in IntelArchive →
                </a>
                {f'&nbsp;·&nbsp;<a href="{link_to_publication}" style="font-size: 0.82em; color: #5cb85c; text-decoration: none; font-family: Arial, sans-serif;">Publication link →</a>' if link_to_publication else ""}
            </div>
            """

    # ── Fetch events data ─────────────────────────────────────────────────────
    events_df      = get_upcoming_events()
    conferences_df = get_upcoming_conferences()
    cfp_df         = get_upcoming_cfp()
    events_html    = build_events_html(events_df, conferences_df, cfp_df)

    html = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #f4f4f4;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f4f4f4;">
            <tr>
                <td align="center" style="padding: 30px 20px;">
                    <table width="640" cellpadding="0" cellspacing="0" style="max-width: 640px; width: 100%; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">

                        <!-- Header -->
                        <tr>
                            <td style="background-color: #1a1a1a; padding: 28px 32px; text-align: left;">
                                <img src="https://raw.githubusercontent.com/yusufaliozkan/zotero-intelligence-bibliography/main/images/01_logo/IntelArchive_Digital_Logo_Colour-Negative.png"
                                     alt="IntelArchive"
                                     width="160"
                                     style="display:block; margin-bottom: 8px;">
                                <p style="color: #aaaaaa; margin: 6px 0 0 0; font-size: 0.85em; font-family: Arial, sans-serif;">
                                    Intelligence Studies Database
                                </p>
                            </td>
                        </tr>

                        <!-- Digest title bar -->
                        <tr>
                            <td style="background-color: #5cb85c; padding: 12px 32px; text-align: left;">
                                <span style="color: #ffffff; font-family: Arial, sans-serif; font-size: 0.95em; font-weight: bold;">
                                    Weekly Digest &nbsp;·&nbsp; {today} &nbsp;·&nbsp; {count} new item{"s" if count != 1 else ""}
                                </span>
                            </td>
                        </tr>

                        <!-- Body: Publications -->
                        <tr>
                            <td style="padding: 28px 32px 0 32px; text-align: left;">
                                <h2 style="font-family: Georgia, serif; color: #1a1a1a; margin: 0 0 8px 0; font-size: 1.3em;">
                                    📚 Recently Added Publications
                                </h2>
                                <p style="font-family: Arial, sans-serif; color: #444; margin: 0 0 20px 0; font-size: 0.95em;">
                                    Here are the latest additions to the
                                    <a href="{BASE_URL}" style="color: #5cb85c; text-decoration: none;">IntelArchive Intelligence Studies Database</a>.
                                </p>
                                {rows_html}
                            </td>
                        </tr>

                        <!-- Body: Events -->
                        {"" if not events_html else f'''
                        <tr>
                            <td style="padding: 0 32px 28px 32px; text-align: left;">
                                <h2 style="font-family: Georgia, serif; color: #1a1a1a; margin: 32px 0 8px 0; font-size: 1.3em;">
                                    📅 Events & Conferences
                                </h2>
                                <p style="font-family: Arial, sans-serif; color: #444; margin: 0 0 20px 0; font-size: 0.95em;">
                                    Upcoming events, conferences, and calls for papers relevant to intelligence studies.
                                    <a href="{EVENTS_URL}" style="color: #5cb85c; text-decoration: none;">See all on IntelArchive →</a>
                                </p>
                                {events_html}
                            </td>
                        </tr>
                        '''}

                        <!-- Footer -->
                        <tr>
                            <td style="background-color: #1a1a1a; padding: 20px 32px; text-align: center;">
                                <p style="font-family: Arial, sans-serif; font-size: 0.78em; color: #888; margin: 0;">
                                    You are receiving this because you are subscribed to the
                                    <a href="https://groups.google.com/g/intelarchive" style="color: #5cb85c; text-decoration: none;">IntelArchive mailing list</a>.
                                </p>
                                <p style="font-family: Arial, sans-serif; font-size: 0.78em; color: #888; margin: 8px 0 0 0;">
                                    <a href="{BASE_URL}" style="color: #5cb85c; text-decoration: none;">Visit IntelArchive</a>
                                    &nbsp;·&nbsp;
                                    <a href="https://bsky.app/profile/intelarchive.io" style="color: #5cb85c; text-decoration: none;">Follow IntelArchive on Bluesky</a>
                                </p>
                            </td>
                        </tr>

                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    return html

def send_digest():
    new_items = get_new_items(days=7)
    count     = len(new_items)

    if count == 0:
        print("No new items this week. Skipping digest.")
        return

    html = build_html_digest(new_items)
    if not html:
        print("Nothing to send.")
        return

    smtp_server   = os.environ["SMTP_SERVER"]
    smtp_port     = int(os.environ["SMTP_PORT"])
    smtp_user     = os.environ["SMTP_USER"]      # ← must be your Gmail address
    smtp_password = os.environ["SMTP_PASSWORD"]  # ← Gmail app password
    to_address    = os.environ["DIGEST_TO"]      # ← Google Group address

    today = datetime.now().strftime("%d %B %Y")
    subject = f"IntelArchive Weekly Digest — {count} new item{'s' if count != 1 else ''} · {today}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"IntelArchive <{smtp_user}>"  # ← shows IntelArchive as sender name
    msg["To"]      = to_address
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_address, msg.as_string())

    print(f"Digest sent: {count} new items to {to_address}")

if __name__ == "__main__":
    send_digest()