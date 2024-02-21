import pandas as pd

def format_entry(row, include_citation=True):
    publication_type = str(row['Publication type']) if pd.notnull(row['Publication type']) else ''
    title = str(row['Title']) if pd.notnull(row['Title']) else ''
    authors = str(row['FirstName2'])
    date_published = str(row['Date published']) if pd.notnull(row['Date published']) else ''
    link_to_publication = str(row['Link to publication']) if pd.notnull(row['Link to publication']) else ''
    zotero_link = str(row['Zotero link']) if pd.notnull(row['Zotero link']) else ''
    published_by_or_in = ''
    published_source = ''
    citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'  
    citation = int(float(citation))
    citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
    citation_link = citation_link.replace('api.', '')

    published_by_or_in_dict = {
        'Journal article': 'Published in',
        'Magazine article': 'Published in',
        'Newspaper article': 'Published in',
        'Book': 'Published by',
    }

    publication_type = row['Publication type']

    published_by_or_in = published_by_or_in_dict.get(publication_type, '')
    published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
    if publication_type == 'Book':
        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
    citation_text = ('Cited by [' + str(citation) + '](' + citation_link + ')' if citation > 0 
        else '')
    return (
        '**' + publication_type + '**' + ': ' +
        title + ' ' +
        '(by ' + '*' + authors + '*' + ') ' +
        '(Publication date: ' + str(date_published) + ') ' +
        ('(' + published_by_or_in + ': ' + '*' + published_source + '*' + ') ' if published_by_or_in else '') +
        '[[Publication link]](' + link_to_publication + ') ' +
        '[[Zotero link]](' + zotero_link + ') ' +
        (citation_text if include_citation else '')
    )