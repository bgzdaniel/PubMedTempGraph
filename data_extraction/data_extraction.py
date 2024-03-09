from extract_utils import fetch_details, get_article_IDs
import pandas as pd
from tqdm import tqdm

def extract_data(extract_params) -> pd.DataFrame:
    """
    Function that calls the Pubmed API via the Entrez package.
    :param
        extract_params:
            hyperparameters specifying start and end date of the data to be extracted
            as well as the time window in which the request is done to receive data in a batch-wise manner.
            If this node fails reduce window duration because request maximum is likely exceeded

    :return:
        pandas dataframe containing the crawled details (abstracts, kewords etc.) of all articles matching the query including.
    """

    print(f"start_date: {extract_params['start_date']}")
    print(f"end_date: {extract_params['end_date']}")

    title_list = []
    authors_list = []
    abstract_list = []
    pubdate_year_list = []

    studiesIdList = get_article_IDs(extract_params) #calls IDs of the articles to fetch detailed data for
    chunk_size = 250  # reduce chunksize to not exceed request limits
    for chunk_i in tqdm(range(0, len(studiesIdList), chunk_size)):
        chunk = studiesIdList[chunk_i:chunk_i + chunk_size]
        papers = fetch_details(chunk)
        for i, paper in enumerate(papers['PubmedArticle']):
            title_list.append(paper['MedlineCitation']['Article']['ArticleTitle'])
            try:
                abstract_list.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
            except:
                abstract_list.append('NA')
            try:
                authors_list.append([", ".join([author.get('LastName'), author.get('ForeName')]) for author in
                                     paper['MedlineCitation']['Article']['AuthorList']])
            except:
                authors_list.append('NA')
            try:
                pubdate_year_list.append(
                    paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'])
            except:
                pubdate_year_list.append('NA')
    df = pd.DataFrame(
            list(zip(
            title_list,
            authors_list,
            abstract_list,
            pubdate_year_list
            )),
            columns=[
                'Title', 'Authors', 'Abstract', 'Year'
            ])
    print("Data extraction finished")
    return df

extract_params = {"window_duration_days": 10,
  "start_date": '2018/01/01',
  "end_date": '2018/01/15'}
studies = extract_data(extract_params)
studies.to_csv("../data/studies.csv", encoding="utf-8", index=False)