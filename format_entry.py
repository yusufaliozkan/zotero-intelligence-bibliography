def format_entry(row, include_citation=True, reviews_map=None, max_reviews_inline=None):
    # Accept Series or dict
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    import pandas as pd

    def _clean(x):
        # Return "" for None/NaN/blank; otherwise stripped string
        try:
            if x is None or pd.isna(x):
                return ""
        except Exception:
            pass
        s = str(x).strip()
        return s if s else ""

    # --- fields ---
    try:
        citation = int(float(row.get("Citation", 0) or row.get("citation_count", 0) or 0))
    except (ValueError, TypeError):
        citation = 0
    citation_link       = _clean(row.get("citation_link"))
    link_to_publication = _clean(row.get("Link to publication"))
    zotero_link         = _clean(row.get("Zotero link"))
    oa_url_fixed        = _clean(row.get("OA_link")).replace(" ", "%20")

    pub_link_badge    = f"[:blue-badge[Publication link]]({link_to_publication})" if link_to_publication else ""
    zotero_link_badge = f"[:blue-badge[Zotero link]]({zotero_link})" if zotero_link else ""
    oa_link_text      = f"[:green-badge[OA version]]({oa_url_fixed})" if oa_url_fixed else ""
    if citation > 0:
        if citation_link:
            citation_text = f"[:orange-badge[Cited by {citation}]]({citation_link})"
        else:
            citation_text = f":orange-badge[Cited by {citation}]"
    else:
        citation_text = ""

    # --- multiple inline review badges ---
    parent_key = row.get("parentKey")
    if not parent_key and zotero_link:
        parent_key = zotero_link.rstrip("/").split("/")[-1]

    book_review_badges = ""
    if reviews_map:
        links = reviews_map.get(parent_key) or []
        if links:
            if len(links) == 1:
                # exactly one review: no number
                book_review_badges = f"[:violet-badge[Book review]]({links[0]})"
            else:
                # multiple reviews: number them
                n = len(links) if max_reviews_inline is None else min(len(links), max_reviews_inline)
                book_review_badges = " ".join(
                    f"[:violet-badge[Book review {i+1}]]({links[i]})" for i in range(n)
                )
                # optional "+N more" cap
                if max_reviews_inline is not None and len(links) > n:
                    # assumes links[0] is newest if you sorted upstream
                    book_review_badges += f" [:violet-badge[+{len(links)-n} more]]({links[0]})"

    # Build the common badges string ONCE
    badges = " ".join(filter(None, [
        pub_link_badge,
        zotero_link_badge,
        book_review_badges,
        oa_link_text,
        citation_text if include_citation else ""
    ]))

    # --- display fields ---
    publication_type = _clean(row.get("Publication type"))
    title           = _clean(row.get("Title"))
    authors         = _clean(row.get("FirstName2"))
    date_published  = _clean(row.get("Date published"))
    book_title      = _clean(row.get("Book_title"))
    thesis_type     = _clean(row.get("Thesis_type"))
    thesis_type2    = f"{thesis_type}: " if thesis_type else ""
    university      = _clean(row.get("University"))

    # NaN-safe journal/publisher logic
    j = _clean(row.get("Journal"))
    p = _clean(row.get("Publisher"))
    if j:
        pub_src_segment = f"(Published in: *{j}*) "
    elif p:
        pub_src_segment = f"(Published by: *{p}*) "
    else:
        pub_src_segment = ""

    # --- output ---
    if publication_type == "Book chapter":
        return (
            f"**{publication_type}**: {title} "
            f"(in: *{book_title}*) "
            f"(by *{authors}*) "
            f"(Publication date: {date_published}) "
            f"{badges}"
        )
    elif publication_type == "Thesis":
        return (
            f"**{publication_type}**: {title} "
            f"({thesis_type2}*{university}*) "
            f"(by *{authors}*) "
            f"(Publication date: {date_published}) "
            f"{badges}"
        )
    else:
        # Books and everything else
        return (
            f"**{publication_type}**: {title} "
            f"(by *{authors}*) "
            f"(Publication date: {date_published}) "
            f"{pub_src_segment}"
            f"{badges}"
        )
