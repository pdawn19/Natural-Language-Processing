#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:24:07 2019

@author: pdawn
"""

'''
This function finds post information like post text, likes, shares, any external links and date.

It stores all this information in a csv file where each row corresponds to one post. 
'''


import numpy as np
import random
from bs4 import BeautifulSoup
import time
#from selenium.common.exceptions import NoSuchElementException
#from webdriverdownloader import geckodriverDownloader
#from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

import helper

def fb_data(html1,profile_ID,profile_url):
    
#    Extract the html with beautiful Soup
    
    soup_data = BeautifulSoup(html1, "html.parser")
    newDiv = soup_data.find_all('div',attrs={'class':"_3ccb"})
#    print(newDiv)

    #each post for keeping a count
    
    post_id = 0
    
    scraped_Data = {} #JSON
    newmatrixcsv = [["Date", "Posts", "Comments", "Likes", "Links", "Views"]]#CSV
    
   

    for num in range(0, len(newDiv)-1):
        data = {}
    
        post_id += 1
        var = "postID" + str(post_id)

        unique_ID = random.randint(1,1001)
        data.update({'_id': unique_ID})
        
#   TimeStamp for the posts
    
        
        date = newDiv[num].find('abbr', attrs={'class': "_5ptz"})
        print(date)

        if date is not None:
            date = date['title']

        else:
            date = ''
        
        data.update({'createdTimeStamp': date})
       
        print('#################################')
        
    
#    Post Description:
        
        div_obj = newDiv[num].find('div', attrs={'class': '_5pbx userContent _3576'})
        text = ''
        if div_obj is not None:
            text += str(helper.clean_text(div_obj.text))
        else:
            text = ''

        data.update({'postDescription': text})
        print(text)
        print('#################################')   
        
#     Post Links:
        
        # external_link = dataDiv[num].find('span', {'class' : 'text_exposed_link'})
        link = newDiv[num].find_all('a', href=True)
        
        linkData = ''
        
        if link is not None:
            for l in link:
                if 'facebook' in l['href']:
                    linkData = linkData + str(l['href'])

                if (str(l['href'])[0] is not '/' and '#' not in l['href'] and 'facebook' not in l['href']):
                    linkData = linkData + str(l['href'])

        print('link', linkData)
        
        print('#################################')
        
        data.update({'postLinks': linkData})
        
#       Comments on Post:
        
        post_allcomment = []

        div_obj = newDiv[num].find('div', attrs={'class': '_3w53'})
        
        comment_id = 0

        if div_obj is not None:               
            for l in div_obj.findAll('div', attrs={'class': '_6qw3'}): 
                comment_details = {'name':[],'comment':[]}
                comment_id +=1
                for n in l.findAll('a', attrs = {'class' : '_6qw4' }):
                    comment_details['name'] = str(n.text)
                for c in l.findAll('span', attrs = {'dir' : 'ltr' }):
                    comment_details['comment']= str(c.text)

#                count= 'CommentID' + str(comment_id)
#                post_allcomment[count] = comment_details
                post_allcomment.append(comment_details)
        

        else:
            continue
              
            
        print(post_allcomment)
        print('#########################')
        
        data.update({'AllComments': post_allcomment})
        
        
#       Likes on Posts:
        
        like = newDiv[num].find('span', {'class': '_81hb'})

        if like is not None:
            like = like.text
        else:
            like = '0'

        likes = like.rsplit(' ')

        like = '0'
        if len(likes) > 0:
            for l in likes:
                if len(l) > 0 and helper.RepresentsInt(l[0]) == True or helper.RepresentsInt(l)==True:
                    like = l
                    if like[len(like)-1] == 'K':
                        like = like[:-1]
                        like = str(int(float(like) * 1000))
                        break
                    break

        print('like', like)
        data.update({'TotalLikes': like})
        print('#####################')
              
              
              
#           Views:
        Views = ''
        Views = newDiv[num].find('div', attr ={'class': '_2ezg'} )
        
        if Views is not None:
                Views = Views.text
                Views = ''
                for i in Views:
                    if i is not ' ':
                        Views += i
                    else:
                        break
        else:
            Views = str(0)
            
#        print(Views)
        data.update({'TotalViews': Views})
        
        
        scraped_Data[var] = data
       
    
#         Write the dictionary in csv document line by line
 
        newrow = [date, text, post_allcomment, like , linkData, Views]
        print(newrow)
        print('##################')   
        newmatrixcsv = np.append(newmatrixcsv, np.array([newrow]), axis=0)

        
    print(newmatrixcsv)
            
#     Write the dictionary in JSON document

    helper.save_file_JSON(scraped_Data)
        
    time.sleep(5)
    print('----------------------------------')
    return newmatrixcsv