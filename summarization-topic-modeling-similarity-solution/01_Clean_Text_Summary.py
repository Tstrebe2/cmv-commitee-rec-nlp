# Databricks notebook source
# pip install -U symspellpy

# COMMAND ----------

# MAGIC %run ./utilities/Data_Clean_Functions

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

# read csv file
reports = pd.read_csv('/dbfs/FileStore/CMV_Reports_Text.csv')
report_dates = pd.read_csv('/dbfs/FileStore/CMV_Report_Dates.csv')

# COMMAND ----------

# reports[reports['PageNumber'] == 890]['ReportText'].tolist()

# COMMAND ----------

# reports.head(10)

# COMMAND ----------

# fill in gaps in PageNumbers
# report_dates

# COMMAND ----------

# reports.sort_values('PageNumber').iloc[-1]['PageNumber']

# COMMAND ----------

# {'PageNumber': reports.sort_values('PageNumber').iloc[-1]['PageNumber'], 'Year': report_dates.iloc[-1]['Year']}

# COMMAND ----------


lastrow = pd.DataFrame({'PageNumber': reports.sort_values('PageNumber').iloc[-1]['PageNumber'], 
                                                      'Year': report_dates.iloc[-1]['Year']}, index=[0])

# COMMAND ----------

report_dates = pd.concat([report_dates, lastrow]).reset_index(drop=True)

# COMMAND ----------

# report_dates

# COMMAND ----------

def create_rows_between(report_dates):
    report_between = pd.DataFrame(columns = ['PageNumber', 'Year'])
    years  = []
    pagenums = []
    for i in range(1, len(report_dates)):
        between_n = report_dates['PageNumber'][i] - report_dates['PageNumber'][i-1] -1
        years.extend(np.repeat([report_dates['Year'][i-1]], between_n ))
        
        prev = report_dates['PageNumber'][i-1]
        for k in range(between_n):
            curr = prev + 1
            pagenums.append(curr)
            prev = curr
            
    report_between['Year'] = years 
    report_between['PageNumber'] = pagenums 
    
    return report_between
report_between = create_rows_between(report_dates)

# COMMAND ----------

# report_between.tail(10)

# COMMAND ----------

report_dates = pd.concat([report_dates, report_between])
# merge report text with dates
reports = reports.merge(report_dates, how = 'outer').reset_index(drop=True)

# COMMAND ----------

reports

# COMMAND ----------

reports['ReportText'] = [remove_extra_space(
    remove_rare_special_character(
        fix_typo(
            remove_lo(
                remove_vertical_line(
                    fix_roma_numbers(
                        remove_extra_dot(
                            remove_hyphen(
                                remove_colon(
                                    remove_quotes(
                                        replace_quotes(
                                            remove_bracket(
                                                normalize_week(
                                                    normalize_month(
                                                        normalize_year(
                                                            i.lower()))))))))))))))).strip() for i in reports['ReportText']]

# COMMAND ----------

TOC_idx = [i for i, s in enumerate(reports['ReportText']) if re.findall("TABLE OF CONTENTs?( ?)+(page)?\n".lower(), s ) ]

# COMMAND ----------

# there should be 23 "table of contents"
reports_TOC = reports.iloc[TOC_idx]
# print(len(reports_TOC))

# COMMAND ----------

# adding table of content page for 2016
reports_TOC = pd.concat([reports_TOC, reports[reports['PageNumber'] == 844]]).reset_index(drop=True)
# now we are only missing 2010 and 2019 which is true in the original doc


# COMMAND ----------

# add table of content manually
reports_TOC = pd.concat([reports_TOC, pd.DataFrame({'PageNumber':190,
'ReportText':'table of contents\nintroduction\ngeneral overview\nReport on the wilmington, de, meeting\ngeneral\nveterans entrepreneurship\nVeterans Entrepreneurship and Small Business Development\nWorkforce Diversity Plan\nGeneral\nSecretary Principis Comments\nVeterans Benefit Clearinghouse VBC \nVocational Rehabilitation & Employment Service\nServing Native American Veterans\nHepatitis C Treatment\nCenter for Minority Veterans\nCommittee Recommendations\nAppendix A\nAppendix B\nAdvisory Committee Agenda Continued\nAppendix C\n'.lower(),
'IsTableOfContents': 1,
'IsCoverPage': 0,
'DocumentBegin': 1,
'IsAttatchment': 0,
'Year': 2001}, index = [0])]).reset_index(drop=True)

# COMMAND ----------

# reports_TOC

# COMMAND ----------

reports_TOC['IsTableOfContents1'] = 1

# COMMAND ----------

reports = reports.merge(reports_TOC[['PageNumber', 'IsTableOfContents1']], how = 'left').reset_index(drop=True)

# COMMAND ----------

# reports['ReportText']

# COMMAND ----------

import math
reports['IsTableOfContents1'] = [0 if math.isnan(i) else int(i) for i in reports['IsTableOfContents1']]

# COMMAND ----------

articleDF = pd.DataFrame(columns = ['Title', 'ArticleText', 'PageNumber'])
for k in range(len(reports_TOC)):
    articleDF = pd.concat([articleDF, separate_text_into_articles(reports_TOC, k)])

# COMMAND ----------

# for k in range(len(reports_TOC)):
k = 20
articleTitles = clean_table_of_content(reports_TOC, k)
report_i = get_report_one_year(reports_TOC, k)
matchedTitle, pageNum, titleStart = find_title(articleTitles, report_i)
[i for i in articleTitles if i not in matchedTitle] 

# COMMAND ----------

articleTitles

# COMMAND ----------

report_i[report_i['PageNumber'] == 975]['ReportText'].tolist()

# COMMAND ----------

# report_i = get_report_one_year(reports_TOC, 22)

# COMMAND ----------

# reports_TOC.loc[k]['Year']

# COMMAND ----------

# separate articles even more by "/n/n" which can be seen as the separation symbol for paragraphs
# first set unique id for each article
# regeneate index for articles
articleDF = articleDF.reset_index(drop = True)
# put index to column
articleDF = articleDF.reset_index(drop = False)
articleDF = articleDF.rename(columns = {'index': 'ArticleIndex'})

# COMMAND ----------

articleDF['Year'] = [reports[reports['PageNumber'] == i[-1]]['Year'].tolist()[0] for i in articleDF['PageNumber']]

# COMMAND ----------

recommd = articleDF[['recommendation' in i for i in articleDF['Title']]]

# COMMAND ----------

recommd.Year.unique()
# missing 2011, 2012, 2003
# there is no recommendation reports in 2003. But there are Issues mentioned there
# fixed 2011 in the raw csv file. table of content part I was scaned as part | ,and in the article it is scaned as l
# fixed 2012. It is similar case with 2011.
# since 2008, the format of recommendation change from "recommendations" only to "recommendations and va responses" to "recommendations, rationales and VA Responses"

# COMMAND ----------

# recommd

# COMMAND ----------

paragraph = []
paragraphNum = []
for i in range(len(recommd)):
    paraghs = recommd.iloc[i]['ArticleText'].split('\n\n')
    paragraph.extend(paraghs)
    paragraphNum.extend(np.repeat(recommd.iloc[i]['ArticleIndex'], len(paraghs)))
paragraphDF = pd.DataFrame({'ParagraphText': paragraph, 'ArticleIndex': paragraphNum})

# COMMAND ----------

# remove newlines in article Text before merge to paragraph
recommd['ArticleText'] = [remove_new_line(i) for i in recommd['ArticleText']]

# COMMAND ----------

# merge paragraph dataframe with article dataframe
recommd = paragraphDF.merge(recommd, how = 'outer')

# COMMAND ----------

recommd['ParagraphText'] = [remove_new_line(i) for i in recommd['ParagraphText']]
# articleDF['ArticleTexts'] = [i.split([' ', '\n']) for i in articleDF['ArticleTexts']]

# COMMAND ----------

recommd

# COMMAND ----------

recommd_temp = recommd[['ParagraphText','ArticleIndex','ArticleText','Year']]

# COMMAND ----------

recommd_temp

# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

