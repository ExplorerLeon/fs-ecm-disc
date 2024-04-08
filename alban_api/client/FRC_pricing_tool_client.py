"""
Created on Wed Feb 22 00:10:22 2017

@author: alban.fauchere
"""

import os
import glob
import sys
import time

import requests
import pandas as pd
import json

from . import configp
from . import inoutput


debug=False


class FRC_api_session(object):
    # settings for session
    def __init__(self):

        self.user = 'insurety'
        self.pwd = '234Nlkd3lsd'
        
        self.processSessionId = 0;
        self.url = "https://long-base-319517.oa.r.appspot.com/pricingTool/api/v1.0"
        
        self.filename = None
        self.filepath = None
        
        # last response
        self.res = None
        # open new session
        self.open_new_session()
    
    
    
    # reset all sessions (admin function)
    def reset_all_sessions(self):
        url = self.url + "/resetAllSessions"
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            print('Reset Open Sessions: ', res_dict)
            return self.res
        except Exception as e:
            print('Session reset failed')
            print(str(e))
	
    
    # check open sessions (admin function)
    def check_open_sessions(self):
        url = self.url + "/checkProcessSessions"
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            print('Open Sessions: ', res_dict)
            return self.res
        except Exception as e:
            print('Session check failed')
            print(str(e))
	
    
    # open session
    def open_new_session(self):
        url = self.url + "/openProcessSession"
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            self.processSessionId = res_dict['processSessionId']
            print('Session', self.processSessionId, 'successfully opened.')
            return self.res
        except Exception as e:
            print('Session opening failed')
            print(str(e))


    # get status
    def get_status(self):
        url = self.url + "/getStatus/" + str(self.processSessionId)
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            print('Process Info: ', res_dict['info'])
            return self.res
        except Exception as e:
            print('Get Status failed - make sure Session has been opened')
            print(str(e))
    
    
    #  upload input data as json to server, import_d is dict of data with tablename as key
    def update_config(self, config_d={}):
        data = json.dumps(config_d)
        # post Json data (sub-set also possible)
        url = self.url + "/loadConfig/" + str(self.processSessionId) + "/json" 
        try:
            self.res = requests.post(url=url, auth=(self.user, self.pwd),
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})
            res_dict = json.loads(self.res.text)
            print('Config update: ', res_dict['status'])
            return self.res
        except Exception as e:
            print('Config update failed - make sure Session is still open')
            print(sys.exc_info())
            if debug:
                tb = sys.exc_info()[2]
                raise e.with_traceback(tb)
    
    
    #  upload input data as json to server, import_d is dict of data with tablename as key
    def upload_import_data(self, import_d={}):
        if import_d.keys():
            data = convert_df_dict_to_json(import_d)
        else:
            print ('Import dict is empty.')
            return
        
        # post Json data
        url = self.url + "/loadData/" + str(self.processSessionId) + "/json/list" 
        try:
            self.res = requests.post(url=url, auth=(self.user, self.pwd),
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})
            res_dict = json.loads(self.res.text)
            print('Data upload: ', res_dict['status'])
            return self.res
        except Exception as e:
            print('Uploading data failed - make sure Session is still open')
            print(sys.exc_info())
            if debug:
                tb = sys.exc_info()[2]
                raise e.with_traceback(tb)
    
    
    # run all
    def run_all(self):
        url = self.url + "/startTask/" + str(self.processSessionId) # + "/startProcessTask"
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            print('Run All: ', res_dict['status'])
            return self.res
        except Exception as e:
            print('Run All failed- make sure Session is still open')
            print(sys.exc_info())
            if debug:
                tb = sys.exc_info()[2]
                raise e.with_traceback(tb)
            
    
    # run all
    def run_pricing_quote(self):
        url = self.url + "/pricingQuote/" + str(self.processSessionId) # + "/startProcessTask"
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            print('Run Pricing Quote: ', res_dict['status'])
            return self.res
        except Exception as e:
            print('Run Pricing Quote failed - make sure Session is still open')
            print(sys.exc_info())
            if debug:
                tb = sys.exc_info()[2]
                raise e.with_traceback(tb)
            
            
    # downloads result data as json from server, and provides a dict of dfs with tablename as key
    def download_result_data(self, report_list=[]):
        if len(report_list) == 0:
            # get all
            report_list = ['Program', 'Portfolio', 'Country_Split', 'Structure', 'SlidingScale', 'RatingToPD', 'CorrMatrix', 'BKG_Ptf', \
                    'Ptf_Stats', 'EL_unlimited_per_CP', 'Layer_Stats', 'Pricing_Quote', 'Prog_Economics', 'Prog_Exposures', 'UW_Account', 'PROG_DECOMP', 'perCP_Metrics']
        #url = self.url + "/getReport/" + str(self.processSessionId) + "/json/" + reportList
        # get all
        url = self.url + "/getReport/" + str(self.processSessionId) + "/json" 
        try:
            self.res = requests.get(url, auth=(self.user, self.pwd))
            res_dict = json.loads(self.res.text)
            print('Result download: ', res_dict['status'])
            return self.res
        except Exception as e:
            print('Result download failed - make sure Session is still open')
            print(sys.exc_info())
            if debug:
                tb = sys.exc_info()[2]
                raise e.with_traceback(tb)
  

 
# import all tables from Excel and creates a dict of pandas dfs
def import_tables_from_Excel(full_filename, config_d={}):
    
    import_d, check_flag = inoutput.import_tables_from_Excel(full_filename, config_d=config_d)
    
    return import_d, check_flag


# applies data check also applied at server level (facultative but allows to check data ahead)
def clean_import_tables(import_d, config_d={}):
    if len(config_d)==0:
        # import from system
        config_d = configp.load_config_parameters()
        
    clean_import_d, inoutput_d, checkflag = inoutput.clean_portfolio_from_settings(import_d, config_d)
    
    return clean_import_d, checkflag
    


# converts dictionary of pandas dataframes to json
def convert_df_dict_to_json(import_d={}):
    data_d = {}
    for tablename in import_d.keys():
        df = import_d[tablename]
        # convert each df to json
        tabledata = df.to_json(orient='split')
        data_d[tablename] = tabledata
    # package in res_dict, other dimensions could be parameters of config
    res_dict = {}
    res_dict['dataList'] = data_d
    # now convert to json
    data_json = json.dumps(res_dict)
    return data_json
    


# converts json reprentation of data to dict of pandas dfs
def convert_json_to_df_dict(data):
    # by convention data tables in attribute 'dataList'
    res_dict = json.loads(data)
    data_d = res_dict['dataList']
    df_d ={}
    for tablename in data_d:
        tabledata = data_d[tablename]  # df in json format
        df = pd.read_json(tabledata, orient='split')
        df_d[tablename] = df
        
    return df_d


# takes result dict of dfs and exports to Excel via user defined format
def export_results_to_Excel(result_filename, df_dict, config_d={}):
    if len(config_d)==0:
        # import from system
        config_d = configp.load_config_parameters()
    inoutput.export_results_to_Excel(result_filename, df_dict, config_d)
    
    
    
# writing output to a new file, only filling tabs where data is not empty
# includes writing out prog_decomp_df
def export_results_to_Excel_raw(result_filename, df_dict):
    
    inoutput.export_results_to_Excel_raw(result_filename, df_dict)



def main():  
    pass






