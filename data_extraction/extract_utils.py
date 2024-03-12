from Bio import Entrez
from datetime import datetime, timedelta
import io
import random
import urllib

def search(query, mindate, maxdate):
    """
    Function to access IDs to queried data using Entrez
    :param
        query: term for which to search
        mindate: end of time period to be extracted
        maxdate: start of time period to be extracteed
    :return:
        Dictionary with the following keys: 'Count', 'RetMax', 'RetStart', 'IdList', 'TranslationSet', 'QueryTranslation'
    """
    #docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
    while True:
        try:
            nums = []
            for _ in range(20):
                nums.append(str(random.randint(0, 9)))
            num = "".join(nums)
            Entrez.email = f'ksdb{num}@emailaddress.com'
            handle = Entrez.esearch(db='pubmed',
                                    sort='relevance',
                                    retmax='10000',
                                    retmode='xml',
                                    term=query,
                                    mindate=mindate,
                                    maxdate=maxdate)
            results = Entrez.read(handle)
            break
        except urllib.error.HTTPError:
            pass
    return results

def fetch_details(id_list) -> dict:
    """
    Function to fetch detailed data for a list of ID's of articles in Pubmed

    :param
        id_list: list of IDs from esearch
    :return:
        nested Dictionary containing the detailed data (in XML)
    """
    ids = ','.join(id_list)

    Entrez.email = 'emails@examples2.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            xml_data = handle.read()
            results = Entrez.read(io.BytesIO(xml_data), validate=False)
            break  # Break out of the loop if successful
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
    else:
        print("Failed after maximum retries.")
    handle.close()
    return results

def get_article_IDs(extract_params) -> list:
    """
    Function that extracts article IDs for later fetching in a batch-wise manner as specified by a window.
    :param
        extract_params:
            Parameters defining start and end date of the query as well as a window iterator
    :return
        list of IDs
    """
    result_dicts = {}
    start_date = datetime.strptime(extract_params['start_date'], '%Y/%m/%d')
    end_date = datetime.strptime(extract_params['end_date'], '%Y/%m/%d')
    window_duration = timedelta(days=extract_params['window_duration_days'])  # timedelta(days=30)
    current_date = start_date
    window_end = start_date + window_duration

    # Loop over time windows of 2 months
    last_iteration = False
    while window_end <= end_date:
        returned_dicts = search('Disease', current_date.strftime('%Y/%m/%d'), window_end.strftime('%Y/%m/%d'))

        # accumulate dictionary values
        for key, value in returned_dicts.items():
            if key in result_dicts:
                if isinstance(value, list):
                    if isinstance(result_dicts[key], list):
                        # If both are lists, extend the existing list with the new list
                        result_dicts[key].extend(value)
                    else:
                        # If the existing value is not a list, create a new list with both values
                        result_dicts[key] = [result_dicts[key]] + value
                else:
                    if isinstance(result_dicts[key], list):
                        # If the existing value is a list, append the new value to it
                        result_dicts[key].append(value)
                    else:
                        # If neither is a list, create a list with both values
                        result_dicts[key] = [result_dicts[key], value]
            else:
                # Add the key-value pair to result_dicts
                result_dicts[key] = value
        current_date = window_end
        window_end = current_date + window_duration
        if last_iteration:
            break
        elif window_end > end_date:
            window_end = end_date
            last_iteration = True

    return result_dicts['IdList']