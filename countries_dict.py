import pandas as pd

df_countries = pd.read_csv('all_items_duplicated.csv')
df_countries = df_countries[df_countries['Collection_Name']=='14 Global intelligence']
country_names = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Côte d'Ivoire", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia",
    "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Holy See", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia",
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
    "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar (formerly Burma)", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Palau", "Palestine State", "Panama", "Papua New Guinea", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia",
    "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States of America", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe", 'Belgium', 'Kosovo', 'Yugoslavia','Mi̇lli̇ İsti̇hbarat Teşki̇latı', 
    'Belgian','Turkish', 'Ottoman Special Organization', 'Belgian', 'British', 'Portuguese', 'Chinese', 'Greek', 'Spanish', 'French', 'Canadian', 'Czechoslovak', 'Soviet','Polish', 'KGB',
    'FSB', 'Dutch', 'German', 'Mossad', 'Norwegian', 'Ottoman', 'Italian', 'Teşkilat-ı Mahsusa', 'Tsar'
]
replacements = {
    'Belgian': 'Belgium',
    'Turkish': 'Turkey',
    'British': 'United Kingdom',
    'Portuguese': 'Portugal',
    'Chinese': 'China',
    'Greek': 'Greece',
    'Spanish': 'Spain',
    'French': 'France',
    'Canadian': 'Canada',
    'Czechoslovak': 'Czechia',
    'Mi̇lli̇ İsti̇hbarat Teşki̇latı': 'Turkey',
    'Soviet': 'Russia',
    'Polish': 'Poland',
    'FSB': 'Russia',
    'Dutch': 'Netherlands',
    'German': 'Germany',
    'Germany':'Germany',
    'Mossad': 'Israel',
    'Norwegian': 'Norway',
    'Ottoman Special Organization':'Turkey',
    'Ottoman':'Turkey',
    'Italian':'Italy',
    'KGB':'Russia',
    'Teşkilat-ı Mahsusa':'Turkey',
    'Tsar':'Russia',
    }

df_countries['Country'] = ''

for country in country_names:
    # Find rows where the Title column contains the country name
    mask = df_countries['Title'].str.contains(country, regex=False)
    
    # Update Country column by concatenating with '|' if multiple countries are found
    df_countries.loc[mask, 'Country'] += country + '|' if not df_countries.loc[mask, 'Country'].empty else ''

# Replace aliases with their respective country names
df_countries['Country'] = df_countries['Country'].str.rstrip('|').replace(replacements, regex=True)
df_countries = df_countries.assign(Country=df_countries['Country'].str.split('|')).explode('Country')
df_countries = df_countries.drop_duplicates(subset=['Country', 'Zotero link'])
df_countries['Country'].replace('', 'Country not known', inplace=True)