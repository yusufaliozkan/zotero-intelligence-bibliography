import requests
from datetime import datetime
import pandas as pd

api_links = [
    'https://api.openalex.org/works?filter=primary_location.source.id:s33269604&sort=publication_year:desc&per_page=10', #IJIC
    "https://api.openalex.org/works?filter=primary_location.source.id:s205284143&sort=publication_year:desc&per_page=10", #The Historical Journal
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210168073&sort=publication_year:desc&per_page=10', #INS
    'https://api.openalex.org/works?filter=primary_location.source.id:s2764506647&sort=publication_year:desc&per_page=10', #Journal of Intelligence History
    'https://api.openalex.org/works?filter=primary_location.source.id:s2764781490&sort=publication_year:desc&per_page=10', #Journal of Policing, Intelligence and Counter Terrorism
    'https://api.openalex.org/works?filter=primary_location.source.id:s93928036&sort=publication_year:desc&per_page=10', #Cold War History
    'https://api.openalex.org/works?filter=primary_location.source.id:s962698607&sort=publication_year:desc&per_page=10', #RUSI Journal
    'https://api.openalex.org/works?filter=primary_location.source.id:s199078552&sort=publication_year:desc&per_page=10', #Journal of Strategic Studies
    'https://api.openalex.org/works?filter=primary_location.source.id:s145781505&sort=publication_year:desc&per_page=10', #War in History
    'https://api.openalex.org/works?filter=primary_location.source.id:s120387555&sort=publication_year:desc&per_page=10', #International History Review
    'https://api.openalex.org/works?filter=primary_location.source.id:s161550498&sort=publication_year:desc&per_page=10', #Journal of Contemporary History
    'https://api.openalex.org/works?filter=primary_location.source.id:s164505828&sort=publication_year:desc&per_page=10', #Middle Eastern Studies
    'https://api.openalex.org/works?filter=primary_location.source.id:s99133842&sort=publication_year:desc&per_page=10', #Diplomacy & Statecraft
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210219209&sort=publication_year:desc&per_page=10', #The international journal of intelligence, security, and public affair
    'https://api.openalex.org/works?filter=primary_location.source.id:s185196701&sort=publication_year:desc&per_page=10',#Cryptologia
    'https://api.openalex.org/works?filter=primary_location.source.id:s157188123&sort=publication_year:desc&per_page=10', #The Journal of Slavic Military Studies
    'https://api.openalex.org/works?filter=primary_location.source.id:s79519963&sort=publication_year:desc&per_page=10',#International Affairs
    'https://api.openalex.org/works?filter=primary_location.source.id:s161027966&sort=publication_year:desc&per_page=10', #Political Science Quarterly
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210201145&sort=publication_year:desc&per_page=10', #Journal of intelligence, conflict and warfare
    'https://api.openalex.org/works?filter=primary_location.source.id:s2764954702&sort=publication_year:desc&per_page=10', #The Journal of Conflict Studies
    'https://api.openalex.org/works?filter=primary_location.source.id:s200077084&sort=publication_year:desc&per_page=10', #Journal of Cold War Studies
    'https://api.openalex.org/works?filter=primary_location.source.id:s27717133&sort=publication_year:desc&per_page=10', #Survival
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210214688&sort=publication_year:desc&per_page=10', #Security and Defence Quarterly
    'https://api.openalex.org/works?filter=primary_location.source.id:s112911512&sort=publication_year:desc&per_page=10', #The Journal of Imperial and Commonwealth History
    'https://api.openalex.org/works?filter=primary_location.source.id:s131264395&sort=publication_year:desc&per_page=10', #Review of International Studies
    'https://api.openalex.org/works?filter=primary_location.source.id:s154084123&sort=publication_year:desc&per_page=10', #Diplomatic History
    'https://api.openalex.org/works?filter=primary_location.source.id:s103350616&sort=publication_year:desc&per_page=10', #Cambridge Review of International Affairs
    'https://api.openalex.org/works?filter=primary_location.source.id:s17185278&sort=publication_year:desc&per_page=10', #Public Policy and Administration
    'https://api.openalex.org/works?filter=primary_location.source.id:s21016770&sort=publication_year:desc&per_page=10', #Armed Forces & Society
    'https://api.openalex.org/works?filter=primary_location.source.id:s41746314&sort=publication_year:desc&per_page=10', #Studies in Conflict & Terrorism
    'https://api.openalex.org/works?filter=primary_location.source.id:s56601287&sort=publication_year:desc&per_page=10', #The English Historical Review
    'https://api.openalex.org/works?filter=primary_location.source.id:s143110675&sort=publication_year:desc&per_page=10', #World Politics
    'https://api.openalex.org/works?filter=primary_location.source.id:s106532728&sort=publication_year:desc&per_page=10', #Israel Affairs
    'https://api.openalex.org/works?filter=primary_location.source.id:s67329160&sort=publication_year:desc&per_page=10', #Australian Journal of International Affairs
    'https://api.openalex.org/works?filter=primary_location.source.id:s49917718&sort=publication_year:desc&per_page=10', #Contemporary British History
    'https://api.openalex.org/works?filter=primary_location.source.id:s8593340&sort=publication_year:desc&per_page=10', #The Historian
    'https://api.openalex.org/works?filter=primary_location.source.id:s161552967&sort=publication_year:desc&per_page=10', #The British Journal of Politics and International Relations
    'https://api.openalex.org/works?filter=primary_location.source.id:s141724154&sort=publication_year:desc&per_page=10', #Terrorism and Political Violence
    'https://api.openalex.org/works?filter=primary_location.source.id:s53578506&sort=publication_year:desc&per_page=10', #Mariner's Mirror
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210184262&sort=publication_year:desc&per_page=10', #Small Wars & Insurgencies
    'https://api.openalex.org/works?filter=primary_location.source.id:s4210236978&sort=publication_year:desc&per_page=10', #Journal of Cyber Policy
    'https://api.openalex.org/works?filter=primary_location.source.id:s120889147&sort=publication_year:desc&per_page=10', #South Asia:Journal of South Asian Studies

    # Add more API links here
]

# Define journals to include filtered items
journals_with_filtered_items = [
    'The Historical Journal', 'Journal of Policing, Intelligence and Counter Terrorism', 'Cold War History', 'RUSI Journal',
    'Journal of Strategic Studies', 'War in History', 'International History Review','Journal of Contemporary History', 
    'Middle Eastern Studies', 'Diplomacy & Statecraft', 'The international journal of intelligence, security, and public affairs',
    'Cryptologia', 'The Journal of Slavic Military Studies', 'International Affairs', 'Political Science Quarterly',
    'Journal of intelligence, conflict and warfare', 'The Journal of Conflict Studies','Journal of Cold War Studies', 'Survival',
    'Security and Defence Quarterly', 'The Journal of Imperial and Commonwealth History', 'Review of International Studies', 'Diplomatic History',
    'Cambridge Review of International Affairs', 'Public Policy and Administration', 'Armed Forces & Society', 'Studies in Conflict & Terrorism',
    'The English Historical Review', 'World Politics', 'Israel Affairs', 'Australian Journal of International Affairs', 'Contemporary British History',
    'The Historian', 'The British Journal of Politics and International Relations', 'Terrorism and Political Violence', "Mariner's Mirror",
    'Small Wars & Insurgencies', 'Journal of Cyber Policy', 'South Asia:Journal of South Asian Studies'
    ]

# Define keywords for filtering
keywords = [
    'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert', 'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis',
    'spying', 'spies'
    ] 

# Initialize an empty list to store DataFrame for each API link
dfs = []

# Loop through each API link
for api_link in api_links:
    # Send a GET request to the API
    response = requests.get(api_link)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the results
        results = data['results']
        
        # Initialize lists to store values for DataFrame
        titles = []
        dois = []
        publication_dates = []
        dois_without_https = []
        journals = []
        
        # Get today's date
        today = datetime.datetime.today().date()
        
        # Extract data for each result
        for result in results:
            # Convert publication date string to datetime object
            pub_date = datetime.datetime.strptime(result['publication_date'], '%Y-%m-%d').date()
            
            # Check if the publication date is within the last # days
            if today - pub_date <= timedelta(days=90):
                titles.append(result['title'])
                dois.append(result['doi'])
                publication_dates.append(result['publication_date'])
                dois_without_https.append(result['ids']['doi'].split("https://doi.org/")[-1])
                journals.append(result['primary_location']['source']['display_name'])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Title': titles,
            'Link': dois,
            'Publication Date': publication_dates,
            'DOI': dois_without_https,
            'Journal': journals
        })
        
        # Append DataFrame to the list
        dfs.append(df)
    
    else:
        print(f"Failed to fetch data from the API: {api_link}")

# Concatenate DataFrames from all API links
final_df = pd.concat(dfs, ignore_index=True)

# Filter 'The Historical Journal' to only include titles containing keywords
historical_journal_filtered = final_df[final_df['Journal'].isin(journals_with_filtered_items)]
historical_journal_filtered = historical_journal_filtered[historical_journal_filtered['Title'].str.lower().str.contains('|'.join(keywords))]

# Filter other journals to exclude 'The Historical Journal'
other_journals = final_df[~final_df['Journal'].isin(journals_with_filtered_items)]

# Concatenate the filtered DataFrames
filtered_final_df = pd.concat([other_journals, historical_journal_filtered], ignore_index=True)