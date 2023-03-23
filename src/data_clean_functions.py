import re
import pandas as pd

def normalize_new_lines(string):
    return "\n".join(string.splitlines())

# COMMAND ----------

def remove_new_line(string):
    return re.sub('(\r)?(\n)+', ' ', string)
# https://note.nkmk.me/en/python-str-replace-translate-re-sub/

# COMMAND ----------

# remove_new_line('adfd \r\n\nfdfd')

# COMMAND ----------

# def remove_digits(string):
#     return re.sub(r'\W[0-9]{1,2}\W', ' ', string)

# COMMAND ----------

def remove_digits(string):
    return re.sub(r'[0-9]+', ' ', string)

# COMMAND ----------

def remove_bracket(string):
    # no removing back slash yet, need that for \n new line
    return re.sub(r'(\[|\]|\(|\)|{|}|\/|<|>|\+|=|!|@|#|\$|%|\^|&|\*|~|`|§)', ' ', string)

# COMMAND ----------

def remove_backslash(string):
    return re.sub(r'\\', '', string)

# COMMAND ----------

# remove_bracket('is [bbb ) department of veterans affairs\ny ry % a : ')

# COMMAND ----------

# def remove_bracket(string):
#     return re.sub(r'(\[|\]|\(|\))', ' ', string)

# COMMAND ----------

def remove_extra_space(string):
    return re.sub(' +', ' ', string)

# COMMAND ----------

def remove_quotes(string):
    return string.replace("'", "").replace('"', '')

# COMMAND ----------

def replace_quotes(string):
    return re.sub('(\’|\“|\”|‘)', "'", string)

# COMMAND ----------

# replace_quotes('“Muggs‘')

# COMMAND ----------

def remove_comma(string):
    return string.replace(',', ' ')

# COMMAND ----------

def remove_colon(string):
    return re.sub(r'\:+', ' ', string)

# COMMAND ----------

def remove_hyphen(string):
    return re.sub(r'(-|—)+', ' ', string)

# COMMAND ----------

# remove_hyphen('part iv agenda - washington')

# COMMAND ----------

def normalize_year(string):
    return re.sub('((199[6-9]|20[01][0-9])|(20[0-2][0-9]))', 'year', string)

# COMMAND ----------

# normalize_year('acmv 2020 date')

# COMMAND ----------

def normalize_month(string):
    return re.sub('( jan | feb | march |april| may | june | july | aug | sep | oct | nov | dec |january|february|august|september|october|november|december)', 'month', string)

# COMMAND ----------

def normalize_week(string):
    return re.sub('( mon | tue | wed | thurs | fri |monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'week', string)

# COMMAND ----------

def normalize_digits(string):
    return re.sub('[0-9]+', 'number', string)

# COMMAND ----------

def remove_vertical_line(string):
    return string.replace('|', ' ')

# COMMAND ----------

# normalize_year('2011')

# COMMAND ----------

def remove_lo(string):
    return string.replace(' lo\n', '\n')

# COMMAND ----------

def remove_extra_dot(string):
    return re.sub(r'''(\.{2,}|\.+[\.,;'" ]\.+)''', '', string)

# COMMAND ----------



# COMMAND ----------

def remove_after_dot(string):
    return re.sub(r'\.+( ?)*(\.|«).*', '', string)

# COMMAND ----------

def remove_rare_special_character(string):
    return re.sub(r'(\^|\*|«|&|%|$|@|`|!|#|\+|=|§)', ' ', string)

# COMMAND ----------

def remove_leading_dot(string):
    return re.sub(r'^( ?)*\.', '', string)

# COMMAND ----------

def fix_typo(string):
    return string.replace('(recommendaation|recommencation)', 'recommendation').replace('introguction', 'introduction').replace('overvicw', 'overview').replace(' dc ', ' d.c. ')

# COMMAND ----------

def remove_errands(string):
    return re.sub(r'(cc( ?)*c| cc | cee | cece).*', '', string)

# COMMAND ----------

def remove_list_symbol(string):
    return re.sub('^ ?([a-z]|[1-9]) ?\.', '', string)

# COMMAND ----------

def remove_appendix(string):
    return re.sub(r'( ?)*(appendix|annex|agenda) [a-h] ?[|, ]+', '', string)

# COMMAND ----------

def remove_unexpected_symbol_end(string):
    return re.sub(r"[:,\-\.\]\)\(\[\{\}'\";<>_+=\|\\&]( ?)*$™", '', string)

# COMMAND ----------

def remove_report_of_the(string):
    return re.sub("^( ?)*report of the ?", '', string)

# COMMAND ----------

# remove_report_of_the('''report of the'''.lower())

# COMMAND ----------

def fix_roma_numbers(string):
    return re.sub('part il\.?', 'part ii',
                  re.sub('part vii\.?', 'part vii', 
                  re.sub('part vi\.?', 'part vi', 
                         re.sub('part v\.?', 'part v', 
                                re.sub('part (i|l)v\.?', 'part iv', 
                                       re.sub('part iii\.?', 'part iii', 
                                              re.sub('part ii\.?', 'part ii', 
                                                     re.sub('part i\.?', 'part i', 
                                                            re.sub('part vil\.?', 'part vii', 
                                                                   re.sub('part v(_|l)\.?', 'part vi', 
                                                                          re.sub('part ill\.?', 'part iii', 
                                                                                 re.sub('part ll\.?', 'part ii', 
                                                                                        re.sub('part!\.?', 'part i', string)))))))))))))

# COMMAND ----------

def remove_roma_part(string):
    return re.sub('part ?(i+|v|i+v|vi+) ', ' ', string)

# COMMAND ----------

def clean_table_of_content(reports_TOC, k):
    articleTitles = remove_digits(reports_TOC['ReportText'][k].lower()).split('\n')
    articleTitles = [remove_extra_space(remove_errands(
        remove_list_symbol(
            remove_report_of_the(
                remove_unexpected_symbol_end(
                    remove_appendix(
                        remove_after_dot(
                            remove_leading_dot(
                                remove_roma_part(i))))))))) for i in articleTitles ]
    articleTitles = [i.strip() for i in articleTitles if i != '']
    articleTitles = [i for i in articleTitles if i != '']
    # remove page
    articleTitles = [i for i in articleTitles if i != 'page']
    return articleTitles

# COMMAND ----------

# remove_extra_space(remove_errands(remove_list_symbol(remove_report_of_the(remove_unexpected_symbol_end(remove_appendix(remove_after_dot(remove_leading_dot('Summary of 2013 Recommendations'))))))))

# COMMAND ----------

# use article titles to find articles and their page numbers
# report_i = reports[(reports['Year'] == reports_TOC.loc[0]['Year']) & (reports['IsTableOfContents1'] == 0)].reset_index(drop=True)
def get_report_one_year(reports, reports_TOC, k):
    return reports[(reports['Year'] == reports_TOC['Year'][k]) ].reset_index(drop=True)

# COMMAND ----------

def find_title(articleTitles, report_i):
    pageNum = []
    titleStart = []
    matchedTitle = []
    # titleText = []
    currPageN = 0
    titlePageN = 0
    for art in articleTitles[1:]:
        pageN = []
        titleS = []
        currPageN = titlePageN 
        while currPageN < len(report_i['ReportText']):
            # if it is content of page Page, then go to the next page
            if (report_i['PageNumber'][currPageN] == report_i[report_i['IsTableOfContents1'] == 1]['PageNumber']).tolist()[0]:
                currPageN += 1
            else:
                # if page number is bigger than the page number from previous title's page number, then continue find this current title's location
                searchPattern = art.lower()+"( ?)*\n"
                searchTarget = report_i['ReportText'][currPageN].lower()
                
#                 if art == 'biographical sketches': #and report_i['PageNumber'][currPageN].tolist() == 144 :
#                     print(report_i['PageNumber'][currPageN].tolist())
                    
#                 print(searchPattern)
                search = re.search(searchPattern, searchTarget)
        #         print('search for '+art+' at page: '+str(report_i['PageNumber'][currPageN]))
                if search is not None:
                    matchedTitle.append(art)
                    pageN.append(report_i['PageNumber'][currPageN])
                    titleS.append(search)
                    titlePageN = currPageN
        #             currPageN += 1
                    break
                else:
                    currPageN += 1

    #     titleText.append(art)
        pageNum.extend(pageN)
    #     pageNum.append(currPageN)
        titleStart.extend(titleS)
    
    # add the start of report and end of report to the list
    matchedTitle = ['Begin of the report'] + matchedTitle
    pageNum = [report_i['PageNumber'][0]] + pageNum
    pageNum = pageNum + [report_i['PageNumber'][len(report_i)-1]]
    titleStart = [0] + titleStart
    # use a very large number to represent the end of the string
    titleStart = titleStart + [10**7]
    # pageNum = [i for i in pageNum if len(i) == 0]
    # titleStart = [i for i in titleStart if len(i) == 0]
    return matchedTitle, pageNum, titleStart

# COMMAND ----------

def match_title_text(pageNum, titleStart, matchedTitle, report_i):
    articleTexts = []
    articlePageNumers = []
    # use page number and title start location to find articles
    for i in range(len(pageNum)-1):
        # if pageNum exist, meaning that there is a title
        # if same page
        articlePageNum = [pageNum[i]]

        if pageNum[i] == pageNum[i+1]:
            articleStart = titleStart[i]
            articleEnd = titleStart[i+1]

            try:
                articleText = report_i[report_i['PageNumber'] == pageNum[i]]['ReportText'].tolist()[0][articleStart.end(): articleEnd.start()]
            except AttributeError:
                articleText = report_i[report_i['PageNumber'] == pageNum[i]]['ReportText'].tolist()[0][articleStart: articleEnd.start()]
        # if different pages
        else:
            articleStart = titleStart[i]
            try:
                articleText = report_i[report_i['PageNumber'] == pageNum[i]]['ReportText'].tolist()[0][articleStart.end():]
            except AttributeError:
                articleText = report_i[report_i['PageNumber'] == pageNum[i]]['ReportText'].tolist()[0][articleStart:]
            # concat string from the next pages
            articlePage = pageNum[i] + 1

            while articlePage < pageNum[i+1]:
                articlePageNum.append(articlePage)
                articleText += '\n'+ report_i[report_i['PageNumber'] == articlePage]['ReportText'].tolist()[0]
                articlePage += 1

            # the end of this article is the start of the next article -> i+1    
            articleEnd = titleStart[i+1]
            try:
                articleText += '\n'+ report_i[report_i['PageNumber'] == pageNum[i+1]]['ReportText'].tolist()[0][:articleEnd.start()]
            except AttributeError:
                articleText += '\n'+ report_i[report_i['PageNumber'] == pageNum[i+1]]['ReportText'].tolist()[0][:articleEnd]
            articlePageNum.append(pageNum[i+1])
        articleTexts.append(articleText)
        articlePageNumers.append(articlePageNum)
    
#     print(len(matchedTitle))
#     print(len(articleTexts))
    cleanedReport_i = pd.DataFrame({
        'Title': matchedTitle,
        'ArticleText': articleTexts,
        'PageNumber': articlePageNumers
    })
    return cleanedReport_i

# COMMAND ----------

def separate_text_into_articles(reports, reports_TOC, k):
    report_i = get_report_one_year(reports, reports_TOC, k)
#     print(len(report_i))
    articleTitles = clean_table_of_content(reports_TOC, k)
#     print(len(articleTitles))
    matchedTitle, pageNum, titleStart = find_title(articleTitles, report_i)
#     print(len(matchedTitle))
#     print(len(pageNum))
#     print(len(titleStart))
    cleanedReport_i = match_title_text(pageNum, titleStart, matchedTitle, report_i)
    return cleanedReport_i

# COMMAND ----------

# pip install -U pip setuptools wheel

# COMMAND ----------

# pip install -U spacy

# COMMAND ----------

# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl

# COMMAND ----------

# import spacy

# COMMAND ----------

# nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

# stops = nlp.Defaults.stop_words

# COMMAND ----------

# def lemmatization(reports):
#     # Tags I want to remove from the text
#     removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
#     tokens = []
#     for summary in nlp.pipe(reports):
#         proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
#         tokens.append(proj_tok)
#     return tokens
