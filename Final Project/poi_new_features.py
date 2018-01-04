# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:54:26 2016

@author: Michael
"""
import pprint

def add_new_features(data_dict):
    
    for i in data_dict:
        if data_dict[i]['total_stock_value'] != 'NaN' and data_dict[i]['total_payments'] != 'NaN':
            summed_total = data_dict[i]['total_stock_value'] + data_dict[i]['total_payments']
            data_dict[i]['total_assets']=summed_total
            #pprint.pprint(data_dict[i]['total_assets'])
        else:
            data_dict[i]['total_assets'] = 'NaN'
            
        if data_dict[i]['from_poi_to_this_person'] != 'NaN' and data_dict[i]['from_this_person_to_poi'] != 'NaN':
            summed_poi_mes = data_dict[i]['from_poi_to_this_person'] + data_dict[i]['from_this_person_to_poi'] 
            data_dict[i]['total_poi_involvement_messages']=summed_poi_mes
            #pprint.pprint(data_dict[i]['total_poi_involvement_messages'])
        else:
            data_dict[i]['total_poi_involvement_messages'] = 'NaN'
    
    return data_dict
    #Total POI interaction