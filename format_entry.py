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
    book_title = ''
    citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'
    citation = int(float(citation))
    citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
    citation_link = citation_link.replace('api.', '')
    thesis_type = str(row['Thesis_type']) if pd.notnull(row['Thesis_type']) else ''
    thesis_type2 = f"{thesis_type}: "
    university = str(row['University']) if pd.notnull(row['University']) else ''

    published_by_or_in_dict = {
        'Journal article': 'Published in',
        'Magazine article': 'Published in',
        'Newspaper article': 'Published in',
        'Book': 'Published by',
    }

    publication_type = row['Publication type']
    published_by_or_in = published_by_or_in_dict.get(publication_type, '')

    if publication_type == 'Journal article' or publication_type == 'Magazine article' or publication_type == 'Newspaper article':
        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
    elif publication_type == 'Book':
        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
    elif publication_type == 'Book chapter':
        book_title = str(row['Book_title']) if pd.notnull(row['Book_title']) else ''

    citation_text = f'Cited by [{citation}]({citation_link})' if citation > 0 else ''
    oa_url = str(row['OA_link']) if pd.notnull(row['OA_link']) else ''
    oa_url_fixed = oa_url.replace(' ', '%20')
    oa_link_text = f'[Open access version]({oa_url_fixed})' if oa_url_fixed else ''

    pub_link_badge = f"[:blue-badge[Publication link]]({link_to_publication})" if link_to_publication else ''
    zotero_link_badge = f"[:blue-badge[Zotero link]]({zotero_link})" if zotero_link else ''
    oa_link_text = f"[:green-badge[OA version]]({oa_url_fixed})" if oa_url_fixed else ''
    citation_text = f"[:orange-badge[Cited by {citation}]]({citation_link})" if citation > 0 else ''


    if publication_type == 'Book chapter':
        return (
            f"**{publication_type}**: {title} "
            f"(in: *{book_title}*) "
            f"(by *{authors}*) "
            f"(Publication date: {date_published}) "
            f"{pub_link_badge} {zotero_link_badge} "
            f"{oa_link_text + ' ' if oa_link_text else ''}"
            f"{citation_text if include_citation else ''}"
        )
    elif publication_type == 'Thesis':
        return (
            f"**{publication_type}**: {title} "
            f"({thesis_type2 if thesis_type != '' else ''}*{university}*) "
            f"(by *{authors}*) "
            f"(Publication date: {date_published}) "
            f"{pub_link_badge} {zotero_link_badge} "
            f"{oa_link_text + ' ' if oa_link_text else ''}"
            f"{citation_text if include_citation else ''}"
        )
    else:
        return (
            f"**{publication_type}**: {title} "
            f"(by *{authors}*) "
            f"(Publication date: {date_published}) "
            f"{f'({published_by_or_in}: *{published_source}*) ' if published_by_or_in else ''}"
            f"{pub_link_badge} {zotero_link_badge} "
            f"{oa_link_text + ' ' if oa_link_text else ''}"
            f"{citation_text if include_citation else ''}"
        )
