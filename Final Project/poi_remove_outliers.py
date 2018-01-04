# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 22:45:02 2016

@author: MC
"""

def remove_outliers(data_dict):
    
#    data_dict_cleaned={}  
    
    to_delete = []
    outliers_to_remove = ['THE TRAVEL AGENCY IN THE PARK','TOTAL']
    #print data_dict
    for i in data_dict:
        
        if i in outliers_to_remove:
            to_delete.append(i)
       
    for i in to_delete:
        data_dict.pop(i,0)
    
    return data_dict
    