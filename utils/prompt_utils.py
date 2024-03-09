

def get_text2date_prompt():
    prompt = """
    Your goal is to accurately extract the list of years explicitly and implicitly mentioned in the user's input. 
    The expected output should be a list of years, capturing both single years and year ranges. If the input contains no year output ['NA'].
    The minimum year is 2013 and the max year is 2023.
    
    Examples:  
    Input: 'Who was CEO at BMW from 1998 to 2003 when the company experienced significant growth?'
    Output: [1998, 1999, 2000, 2001, 2002, 2003]
    
    Input: 'Cancer 2011-2023'
    Output: [2011, 2012, 2013, 2014, 2ß15, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    
    Input: 'What research was published on Dysarthria in 2010 and 2015?
    Output: [2010, 2015]
    
    Input: 'Was there a major terrorist attack in 2009?'
    Output: [2009]
    
    Input: 'Who won the worldcup in 2012?'
    Output: [2012]
    
    Input: 'What research was published on Covid after 2018 but before 2023?'
    Output: [2019, 2020, 2021, 2022]

    Input: 'What research was published on Covid since 2018?'
    Output: [2018, 2019, 2020, 2021, 2022, 2023]
    
    Input: 'Examine breakthroughs in medical research between 2010 and 2015.'
    Output: [2010, 2011, 2012, 2013, 2014, 2015]
    
    Input: 'What is Autism?'
    Output: ['NA']
    
    ______
    Do NOT output anything else but the list of years. Do NOT provide additional information.
    Ensure your output follows this format "[year_x, year_y, .., year_z]".
    
    User Input: {question}
    Your Output:
    """
    return prompt

def get_overview_prompt():
    prompt = """
    Take the provided published papers from PubMed and summarize the abstracts into one summary. 
    Start the summary with the year of the papers. All provided papers will be from the same year. 
    Keep your summary to three sentences at maximum. Do NOT exceed the max sentence amount. 
    Do not reference the authors in your summary. Mention the first authors at the end of the summary.
    Remember to summarize all papers into only one summary!
    Here are the published papers: '{context}' 
    Your output: 
    """
    return prompt

def get_research_prompt():
    prompt = """
    You are a biomedical AI assistant to answer medical questions
    mostly about PubMed articles provided as context for you.
    Not every article in the provided context is necessarily relevant to the question.
    Carefully examine the provided information in the articles and choose the
    most likely correct information to answer the question.
    If the question is not from the biomedical domain, tell the user that
    the question is out of domain and cannot be answered by you.
    As an AI assistant, answer the question accurately,
    precisely and concisely. Only include information in your answer,
    which is necessary to answer the question.
    Be as short and concise in your answer as possible.
    Do NOT mention that your answer is based on the provided paper or context.
    
    Use the following articles to determine the answer: {context}
    The question: {question}
    Your answer:
'''

    """
    return prompt