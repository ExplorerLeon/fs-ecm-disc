# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 08:45:42 2017

@author: alban
"""

import os
import time
#import xml.etree.ElementTree as ET
import xml.etree.cElementTree as ET

import pickle
import numpy as np
import pandas as pd

config_fname = 'alban_api/system/config.xml'
settings_fname = 'alban_api/system/settings.xml'


def load_config_parameters(config_file=config_fname, cnt=True):
    #here = os.path.dirname(os.path.abspath(__file__))
    #path = os.path.join(here,config_fname)
    config_d = {}
    tree = ET.parse(config_file)
    root = tree.getroot()
    # import config parameters
    config_d['tid'] = int(root.find('tid').text)
    config_d['seed_flag'] = bool(float(root.find('seed_flag').text))
    config_d['ver'] = float(root.find('ver').text)
    config_d['client_spec_flag'] = int(root.find('client_spec_flag').text)
    config_d['image_file'] = str(root.find('image_file').text)
    config_d['num_scenarios'] = int(root.find('num_scenarios').text)
    config_d['max_tenor'] = int(root.find('max_tenor').text)
    config_d['bin_thres'] = int(root.find('bin_thres').text)
    config_d['lnl_thres'] = int(root.find('lnl_thres').text)
    #config_d['num_sev_samples'] = int(root.find('num_sev_samples').text)
    config_d['mf_flag'] = bool(float(root.find('mf_flag').text))
    config_d['bkg_epi'] = float(root.find('bkg_epi').text)
    # expected loss multiple and return on capacity
    config_d['def_rating'] = str(root.find('def_rating').text)
    config_d['named_R'] = float(root.find('named_R').text)
    config_d['unnamed_R'] = float(root.find('unnamed_R').text)
    config_d['shape'] = float(root.find('shape').text)
    config_d['expense_ratio'] = float(root.find('expense_ratio').text)
    config_d['elm'] = float(root.find('elm').text)
    config_d['roc'] = float(root.find('roc').text)
    config_d['coc'] = float(root.find('coc').text)
    config_d['target_return'] = 0
    config_d['return_type'] = 'target_uw_margin'
    #config_d['gu_decomp_ptf'] = str(root.find('gu_decomp_ptf').text)
    # settings
    config_d['block_flag'] = bool(float(root.find('block_flag').text))
    config_d['block_size'] = int(root.find('block_size').text)
    
    config_d['group_by_cp_flag'] = bool(float(root.find('group_by_cp_flag').text))
    config_d['severity_by_policy_flag'] = bool(float(root.find('severity_by_policy_flag').text))
    config_d['exp_struct_flag'] = bool(float(root.find('exp_struct_flag').text))
    config_d['xl_on_qs_ret_flag'] = bool(float(root.find('xl_on_qs_ret_flag').text))
    config_d['type_sub_tot_flag'] = bool(float(root.find('type_sub_tot_flag').text))
    config_d['gross_net_econ_flag'] = bool(float(root.find('gross_net_econ_flag').text))
    # int counting number of years
    config_d['uw_year_split_flag'] = min(15, max(0, int(root.find('uw_year_split_flag').text) ))
    config_d['reinst_flag'] = bool(float(root.find('reinst_flag').text))
    config_d['slide_flag'] = bool(float(root.find('slide_flag').text))
    config_d['pc_slide_flag'] = bool(float(root.find('pc_slide_flag').text))
    config_d['cession_flag'] = bool(float(root.find('cession_flag').text))
    config_d['ccy_flag'] = bool(float(root.find('ccy_flag').text))
    config_d['share_flag'] = bool(float(root.find('share_flag').text))
    config_d['print_flag'] = bool(float(root.find('print_flag').text))
    config_d['output_flag'] = bool(float(root.find('output_flag').text))
    config_d['decomp_flag'] = bool(float(root.find('decomp_flag').text))
    config_d['file_flag'] = True
    config_d['debug_flag'] = bool(float(root.find('debug_flag').text))
    config_d['graph_flag'] = bool(float(root.find('graph_flag').text))
    config_d['ULSC_flag'] = bool(float(root.find('ULSC_flag').text))
    config_d['color_code_flag'] = bool(float(root.find('color_code_flag').text))
    config_d['limit_per_name_flag'] = bool(float(root.find('limit_per_name_flag').text))
    config_d['uw_summary_flag'] = bool(float(root.find('uw_summary_flag').text))
    config_d['run_off_flag'] = bool(float(root.find('run_off_flag').text))
    config_d['cntry_split_flag'] = bool(float(root.find('cntry_split_flag').text))
    
    size_bins_list = [-np.inf]
    size_bins_list.extend([float(item) for item in root.find('size_bins').text.split(',') ])
    size_bins_list.append(np.inf)
    config_d['size_bins'] = size_bins_list
    config_d['named_bin_flag'] = bool(float(root.find('named_bin_flag').text))
    config_d['max_named_bin'] = float(root.find('max_named_bin').text)
    
    config_d['per_sc_per_cp_output_flag'] = bool(float(root.find('per_sc_per_cp_output_flag').text))
    config_d['per_cp_metrics_flag'] = bool(float(root.find('per_cp_metrics_flag').text))
    if not config_d['gross_net_econ_flag']:
        config_d ['per_cp_metrics_flag'] = False
    
    
    return config_d




    
# set client specific setting input / output fields
# note that this should cater for all config_d settings (which may be overriden by Program tab)
def define_inoutput_d(config_d, prog_df=pd.DataFrame()): 
    
    def float_sep(val):
        if np.isnan(val):
            return ''
        else:
            return '{:,.0f}'.format(val)
    def float_dec(val):
        if np.isnan(val):
            return ''
        else:
            return '{:,.2f}'.format(val)
    def perc_high(val):
        if np.isnan(val):
            return ''
        else:
            return '{:.2%}'.format(val)
    def perc_med(val):
        if np.isnan(val):
            return ''
        else:
            return '{:.1%}'.format(val)
    def perc_low(val):
        if np.isnan(val):
            return ''
        else:
            return '{:.0%}'.format(val)    
    
    inoutput_d = {}
    
    name = 'Program'
    settings_d = {}
    
    def_rating = config_d['def_rating']
    expense_ratio = config_d['expense_ratio']
    exp_struct_flag = config_d['exp_struct_flag']
    xl_on_qs_ret_flag = config_d['xl_on_qs_ret_flag']
    cession_flag = config_d['cession_flag']
    color_code_flag = config_d['color_code_flag']
    group_by_cp_flag = config_d['group_by_cp_flag']
    severity_by_policy_flag = config_d['severity_by_policy_flag']
    uw_year_split_flag = config_d['uw_year_split_flag']
    limit_per_name_flag = config_d['limit_per_name_flag']
    run_off_flag = config_d['run_off_flag']
    named_bin_flag = config_d['named_bin_flag']
    
    settings_d['key_cols'] = ['Cedent', 'UW Year', 'Currency', 'ExRate']
    settings_d['rel_cols_with_dfval'] = {'Contract_ID': '', 'Legal Carrier': '', 'Program': 'None', 'Type': 'None', 'CType': 'TBD', 'Country': 'None', 'LOB': 'None', 'ExRate': 1.0, 'Def Rating': def_rating, 'Growth factor': 1.0, 'Expense Ratio': expense_ratio, \
              'Exp Struct': exp_struct_flag, 'XL on QS Ret': xl_on_qs_ret_flag, 'Group by CP': group_by_cp_flag, 'Severity by Policy': severity_by_policy_flag, 'UW Year Split': uw_year_split_flag, 'Exp Run Off': run_off_flag, 'Auto Bands': named_bin_flag}
    settings_d['int_cols'] = ['UW Year']
    settings_d['float_cols'] = ['ExRate', 'Growth factor']
    # formatting ouput
    col_order = ['Contract_ID', 'Legal Carrier', 'Cedent', 'Program', 'Type', 'LOB', 'UW Year', 'Country', 'Currency', 'ExRate', 'Def Rating', 'Growth factor']
    if config_d['gross_net_econ_flag']:
        col_order.append('Expense Ratio')
    # override config_d setting with prog_df
    if prog_df.empty:
        if config_d['exp_struct_flag']:
            col_order.extend(['Exp Struct'])
        if config_d['xl_on_qs_ret_flag']:
            col_order.extend(['XL on QS Ret'])
        if config_d['group_by_cp_flag']:
            col_order.extend(['Group by CP'])
        if config_d['severity_by_policy_flag']:
            col_order.extend(['Severity by Policy'])
        if config_d['uw_year_split_flag']:
            col_order.extend(['UW Year Split'])
        if config_d['run_off_flag']:
            col_order.extend(['Exp Run Off'])
        if config_d['named_bin_flag']:
            col_order.extend(['Auto Bands'])
    else:
        if prog_df['Exp Struct'].max():
            col_order.extend(['Exp Struct'])
        if prog_df['XL on QS Ret'].max():
            col_order.extend(['XL on QS Ret'])
        if prog_df['Group by CP'].max():
            col_order.extend(['Group by CP'])
        if prog_df['Severity by Policy'].max():
            col_order.extend(['Severity by Policy'])
        if prog_df['UW Year Split'].max():
            col_order.extend(['UW Year Split'])
        if prog_df['Exp Run Off'].max():
            col_order.extend(['Exp Run Off'])
        if prog_df['Auto Bands'].max():
            col_order.extend(['Auto Bands'])
    
    settings_d['col_order'] = col_order
    col_width_d = {}
    col_width_d['A:B'] = 11
    col_width_d['C:D'] = 20
    col_width_d['I:I'] = 15
    col_width_d['L:S'] = 12
    settings_d['col_width_d'] = col_width_d
    
    settings_d['color_code'] = color_code_flag
    col_format_dict = {}
    if color_code_flag:
        col_format_dict['green'] = ['Cedent', 'UW Year', 'Currency', 'ExRate']   # key cols
        col_format_dict['orange'] = ['Exp Struct', 'XL on QS Ret', 'Group by CP', 'Severity by Policy']
        # all otther blue
        col_format_dict['wide'] = ['UW Year', 'Country', 'Def Rating', 'Growth factor', 'Expense Ratio' 'Exp Struct', 'XL on QS Ret']
        col_format_dict['xwide'] = ['Cedent', 'Program']
        # all other narrow
        col_format_dict['fmt_decimal'] = ['ExRate', 'Expense Ratio', 'Growth factor']
        settings_d['col_format_dict'] = col_format_dict
    
    inoutput_d[name] = settings_d

    if prog_df.empty:
        # return only Program settings
        return inoutput_d
        
        
    name = 'Portfolio'
    settings_d = {}
    
    def_rating = prog_df['Def Rating'].max()
    def_country = prog_df['Country'].max()
    def_lob = prog_df['LOB'].max()
    def_R = config_d['named_R']
    def_shape = config_d['shape']
    if prog_df['XL on QS Ret'].max():
        settings_d['key_cols'] = ['Company Name', 'Notional', 'PEL']
    else:
        settings_d['key_cols'] = ['Company Name', 'Notional Retained', 'Notional Ceded', 'PEL']
    # note notional = ceded + retained is not checked at import but when applying cession
    rel_cols_with_dfval =  {'Notional': 0.0, 'Notional Retained': 0.0, 'Notional Ceded': 0.0, 'Country': def_country, 'LOB': def_lob, 'RiskFactor': 0, 'Rating': def_rating, 'Cardinality': 1, 'LNL' : 0, 'Tenor': 1, 'PML': 0.9, 'PEL': 0.5, 'Shape': def_shape, 'R': def_R, 'Attach': 0.0, 'Detach': 1.0, 'ORIG_CP_NAME': ''}
    int_cols = ['RiskFactor', 'Cardinality']
    float_cols = ['Notional', 'Notional Retained', 'Notional Ceded', 'Tenor','PEL', 'PML', 'Shape', 'R', 'Attach', 'Detach']
    if limit_per_name_flag:
        rel_cols_with_dfval['Limit'] = 0.0
        float_cols.append('Limit')
    # uw year split
    if prog_df['UW Year Split'].max():  #config_d['uw_year_split_flag']:
        uy_cols = [ ('UY-'+ str(i)) for i in range(prog_df['UW Year Split'].max())]
        for uwy in uy_cols:
            rel_cols_with_dfval[uwy] = 0.0
        rel_cols_with_dfval['UY-0'] = 1.0
        float_cols.extend(uy_cols)
    settings_d['rel_cols_with_dfval'] = rel_cols_with_dfval
    settings_d['int_cols'] = int_cols
    settings_d['float_cols'] = float_cols
    
    # formatting output (no Risk Factor)
    col_order = ['Company Name', 'Country', 'LOB', 'Rating', 'PD', 'R']
    if config_d['mf_flag']:
        col_order.extend(['RiskFactor'])
    col_order.extend(['Cardinality', 'LNL'])
    if limit_per_name_flag:
        col_order.extend(['Limit'])
    col_order.extend(['Notional', 'Notional Ceded', 'Notional Retained', 'Tenor', 'PEL', 'PML', 'Shape', 'PML_90th'])
    if prog_df['Exp Struct'].max():
        col_order.extend(['Attach', 'Detach'])
    if prog_df['UW Year Split'].max():
        col_order.extend(uy_cols)
    col_order.extend(['ORIG_CP_NAME'])
    settings_d['col_order'] = col_order
    col_width_d = {}
    col_width_d['A:A'] = 25
    if config_d['mf_flag']:
        if limit_per_name_flag:
            col_width_d['J:M'] = (15, 'fmt_numbers')
        else:
            col_width_d['J:L'] = (15, 'fmt_numbers')
    else:
        if limit_per_name_flag:
            col_width_d['I:L'] = (15, 'fmt_numbers')
        else:
            col_width_d['I:K'] = (15, 'fmt_numbers')
    settings_d['col_width_d'] = col_width_d
    inoutput_d[name] = settings_d
    # could allow for PD to be entered directly
    
    name = 'BKG_Ptf'
    settings_d = {}
    settings_d['key_cols'] = ['Company Name', 'Notional', 'Rating','PEL']
    settings_d['rel_cols_with_dfval'] = {'Notional': 0.0, 'Country': 'WW', 'LOB': '', 'RiskFactor': 0, 'Cardinality': 1, 'LNL' : 0, 'Tenor': 1, 'PML': 0.9, 'Shape': 4, 'R': 0.2}
    settings_d['int_cols'] = ['RiskFactor', 'Cardinality']
    settings_d['float_cols'] = ['Notional', 'Tenor','PEL', 'PML', 'Shape', 'R']
    # no output
    inoutput_d[name] = settings_d
    
    name = 'RatingToPD'
    settings_d = {}
    settings_d['key_cols'] = ['Rating', 'PD']
    settings_d['rel_cols_with_dfval'] = {}
    settings_d['int_cols'] = []
    settings_d['float_cols'] = ['PD']
    # no output
    inoutput_d[name] = settings_d
     
    name = 'CorrMatrix'
    settings_d = {}
    settings_d['key_cols'] = ['RF 0', 'RF 1', 'RF 2', 'RF 3']
    settings_d['rel_cols_with_dfval'] = {}
    settings_d['int_cols'] = []
    settings_d['float_cols'] = ['RF 0', 'RF 1', 'RF 2', 'RF 3']
    # no output, for debug only
    settings_d['show'] = False
    inoutput_d[name] = settings_d
    
    name = 'Structure'
    settings_d = {}
    settings_d['key_cols'] = ['Type', 'Attach', 'Limit', 'AggAttach', 'AggLimit', 'EPI', 'rate', 'Commission', 'Brokerage', 'Tax', 'No Claims Bonus', 'Profit Comm', 'Mgmt Exp', 'ReinstPrem']      
    # prepare settings
    rel_cols_with_dfval = {'Layer_ID': '', 'Attach': 0.0, 'Limit': 0.0, 'AggAttach': 0.0, 'AggLimit': np.nan, 'Num Reinst': 0, 'rate': 1.0, 'Commission': 0.0, 'Brokerage': 0.0, 'Tax': 0.0, 'No Claims Bonus': 0.0, 'Profit Comm': 0.0, 'Mgmt Exp': 0.0, 'ReinstPrem': 0.0, 'Interlocking': 0, 'Share': 1.0}
    int_cols = []
    float_cols = ['Attach', 'Limit', 'AggAttach', 'AggLimit', 'Num Reinst', 'EPI', 'rate', 'Commission', 'Brokerage', 'Tax', 'No Claims Bonus', 'Profit Comm', 'Mgmt Exp', 'ReinstPrem', 'Share', 'Interlocking']
    if limit_per_name_flag:
        rel_cols_with_dfval['AutoLimit'] = 0.0
        float_cols.append('AutoLimit')
    if prog_df['XL on QS Ret'].max() or config_d['cession_flag']:
        rel_cols_with_dfval['Cession'] = 1.0
        float_cols.extend(['Cession'])
    if config_d['reinst_flag']:
        rel_cols_with_dfval['ReinstPrem2'] = 0
        rel_cols_with_dfval['ReinstPrem3'] = 0
        rel_cols_with_dfval['ReinstPrem4'] = 0
        rel_cols_with_dfval['ReinstPrem5'] = 0
        float_cols.extend(['ReinstPrem2', 'ReinstPrem3', 'ReinstPrem4', 'ReinstPrem5'])
    if config_d['slide_flag']:
        rel_cols_with_dfval['SlidingScale'] = 0
        rel_cols_with_dfval['SS_ID'] = 0
        int_cols.extend(['SlidingScale'])
    if prog_df['UW Year Split'].max():  #config_d['uw_year_split_flag']:
        suy_cols = [ ('Share-'+ str(i)) for i in range(prog_df['UW Year Split'].max())]
        for suy in suy_cols:
            rel_cols_with_dfval[suy] = 0
        rel_cols_with_dfval['Share-0'] = 1.0
        float_cols.extend(suy_cols)
    settings_d['rel_cols_with_dfval'] = rel_cols_with_dfval
    settings_d['int_cols'] = int_cols
    settings_d['float_cols'] = float_cols
    # foramtting output
    col_order = ['Layer_ID', 'Type']
    if prog_df['XL on QS Ret'].max() or config_d['cession_flag']:
        col_order.extend(['Cession'])
    col_order.extend(['Share', 'Attach', 'Limit', 'AggAttach', 'AggLimit', 'EPI', 'rate', 'Commission', 'Brokerage', 'Tax', 'No Claims Bonus', 'Profit Comm', 'Mgmt Exp', 'ReinstPrem'])
    if config_d['reinst_flag']:
        col_order.extend(['ReinstPrem2', 'ReinstPrem3', 'ReinstPrem4', 'ReinstPrem5'])
    col_order.extend(['Num Reinst', 'Interlocking'])
    if config_d['slide_flag']:
        col_order.extend(['SlidingScale', 'SS_ID'])
    if limit_per_name_flag:
        col_order.append('AutoLimit')
    if prog_df['UW Year Split'].max():
        suy_cols = [ ('Share-'+ str(i)) for i in range(prog_df['UW Year Split'].max())]
        col_order.extend(suy_cols)
    settings_d['col_order'] = col_order
    col_width_d = {}
    if prog_df['XL on QS Ret'].max() or config_d['cession_flag']:
        col_width_d['D:E'] = ( 8, 'fmt_perc')
        col_width_d['F:J'] = ( 15, 'fmt_numbers')
        col_width_d['K:K'] = ( 8, 'fmt_rate')
        col_width_d['L:V'] = ( 8, 'fmt_perc')
        if limit_per_name_flag:
            col_width_d['AA:AA'] = ( 15, 'fmt_numbers')
    else:
        col_width_d['D:D'] = ( 8, 'fmt_perc')
        col_width_d['E:I'] = ( 15, 'fmt_numbers')
        col_width_d['J:J'] = ( 8, 'fmt_rate')
        col_width_d['K:U'] = ( 8, 'fmt_perc')
        if limit_per_name_flag:
            col_width_d['Z:Z'] = ( 15, 'fmt_numbers')
    settings_d['col_width_d'] = col_width_d
    inoutput_d[name] = settings_d
    
    name = 'SlidingScale'
    settings_d = {}
    if config_d['pc_slide_flag']:
        key_cols = ['SS_ID', 'LossRatio', 'CommRatio']      
        float_cols = ['LossRatio', 'CommRatio', 'PCRatio']
    else:
        key_cols = ['SS_ID', 'LossRatio', 'CommRatio']      
        float_cols = ['LossRatio', 'CommRatio']
    settings_d['key_cols'] = key_cols 
    settings_d['int_cols'] = []
    settings_d['float_cols'] = float_cols
    settings_d['rel_cols_with_dfval'] = {'PCRatio': 0}
    # fomatting output
    col_order = ['SS_ID', 'LossRatio', 'CommRatio']
    if config_d['pc_slide_flag']:
        col_order.append('PCRatio')
    settings_d['col_order'] = col_order
    col_width_d = {}
    col_width_d['B:D'] = ( 15, 'fmt_perc')
    settings_d['col_width_d'] = col_width_d
    inoutput_d[name] = settings_d
    
    name = 'Ptf_Stats'
    settings_d = {}
    # output only
    settings_d['ptf_rename'] = {'QS': 'Ceded to QS', 'XL': 'Retained to XL', 'TOT': 'Total Ptf'}
    settings_d['col_order'] = ['Notional', 'Cardinality', 'Avg_Notl', 'Max_Notl', r'Notl_Named%', 'PD', 'PEL', 'PML', 'EL', 'EPI', 'Prem_Rate', 'LR', 'LR_Named']
    col_order_ext = list(settings_d['col_order'])
    #col_order_ext.extend(['Expected Loss', 'Stdv Loss', 'Median Loss', '10Y Loss', '50Y Loss',  '100Y Loss', '250Y Loss', 'SA SF Loss', 'Div SF Loss', 'Stdv LR', 'SA SF LR', 'Div SF LR', 'Hit rate', 'Frequency_stat', 'Frequency', 'Loss Severity'])
    col_order_ext.extend(['Expected Loss', 'Stdv Loss', 'Median Loss',  '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '200Y SF Loss', '250Y SF Loss', 'SA SF Loss', 'Div SF Loss', 'Stdv LR', '10Y SF LR','50Y SF LR',  '100Y SF LR', '200Y SF LR', '250Y SF LR', 'SA SF LR', 'Div SF LR', 'Hit rate', 'Frequency', 'Loss Severity'])
    settings_d ['col_order_ext'] = col_order_ext
    # display table for ground up portfolio stats
    def set_column_settings_for_ptf_stats():
        col_order_dict =      [ 'EPI',   'Prem_Rate',  'LR',    'LR_Named',  'PD',    'PEL',   'Notl_Named%', 'Cardinality' , 'Avg_Notl',   'Frequency', 'EL',      'Max_Notl',  'PML',   'Max_PML_Notl']
        col_string_converter = [float_sep,  perc_high, perc_low, perc_low,  perc_high,  perc_low, perc_low,    float_sep,  float_sep,       float_dec,    float_sep, float_sep,  perc_low, float_sep]
        col_width =            [ 90,         100,        55,        95,      55,       60 ,         115,            75,        90,             90,            90,        120,        65,      130]
        #col_order_dict =      [ 'EPI', 'Prem_Rate', 'Cardinality' , 'Avg_Notl', 'PD',    'PEL',    'PML',   'LR',    'LR_Named',  'EL',  'Frequency',  'Notl_Named%', 'Max_Notl',  'Max_PML_Notl',  'Notional']
        #col_string_converter = [float_sep,  perc_high, float_sep,  float_sep, perc_high, perc_low, perc_low, perc_low, perc_low,   float_sep, float_dec,  perc_low,   float_sep,    float_sep,  float_sep]
        #col_width =            [ 90,     70,        70,             70,        55,        50,       50,     50,        70,         75,       70,          70,            90,         90,            110]
        return col_order_dict, col_string_converter, col_width
    settings_d['olv_setting_function'] = set_column_settings_for_ptf_stats
    inoutput_d[name] = settings_d
    
    name= 'EL_unlimited_per_CP'
    settings_d = {}
    settings_d['show'] = True
    inoutput_d[name] = settings_d
    
    name = 'Exposure Stats'
    settings_d = {}
    # display table for struct exposure stats
    def set_column_settings_for_struct_exp_stats():
        col_order_dict =      [ 'Type', 'Limit', 'rate', '# Notl',    'Notl%', '# PML',  'PEL_Notl%', 'EL_gu%',  'EPI',     'EL_unlimited',      'RoL', 'Attach_Freq', 'Detach_Freq']
        col_string_converter = ['%s', float_sep, perc_high, float_sep, perc_med, float_sep, perc_med, perc_high,  float_sep,    float_sep,  perc_med, float_dec,  float_dec]
        col_width =            [ 40,     85,      65,      75,          75,       75,        85,        85,        100,        100,            65,     90,         90]
        #col_order_dict =      [ 'Type', 'Limit', '# Notl', 'Notl%', '# PML',  'PEL_Notl%', 'EL_gu%',  'EL_unlimited',  'EPI',     'rate',  'RoL', 'Attach_Freq', 'Detach_Freq']
        #col_string_converter = ['%s', float_sep, float_sep, perc_med, float_sep, perc_med, perc_high,  float_sep,    float_sep, perc_high,  perc_med, float_dec,  float_dec]
        #col_width =            [ 50,     90,        60,      60,       60,        65,      60,        80,             80,         60,       60,     80,           80]
        return col_order_dict, col_string_converter, col_width 
    settings_d['olv_setting_function'] = set_column_settings_for_struct_exp_stats
    inoutput_d[name] = settings_d
    
    name = 'Ground-up Stats'
    settings_d = {}
    # display table for ground up portfolio stats
    def set_column_settings_for_gup_stats():
        col_order_dict =      [ 'LR',    'Stdv LR',  '10Y SF LR', 'SA SF LR', 'Div SF LR', 'Hit rate', 'Premium', 'Expected Loss', 'Stdv Loss', 'Median Loss', '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '250Y SF Loss', 'SA SF Loss', 'Div SF Loss']
        col_string_converter = [perc_low, perc_low,  perc_low,    perc_low, perc_low,      perc_med, float_sep,    float_sep,      float_sep,  float_sep,      float_sep,      float_sep,     float_sep,     float_sep,      float_sep,    float_sep]
        col_width =            [ 60,     80,        90,            90,             90,        80,             100,       105,               90,          95,         95,           95,              100,         100,            100,           100]
        return col_order_dict, col_string_converter, col_width
    settings_d['olv_setting_function'] = set_column_settings_for_gup_stats
    inoutput_d[name] = settings_d
    
    name = 'Layer_Stats'
    settings_d = {}
    # output only
    settings_d['col_order'] = ['Layer_ID', 'Type', 'Attach', 'Limit', 'AggAttach', 'AggLimit','# Notl', 'Notl%', '# PML', 'PEL_Notl%', 'EL_unlimited', 'EL_gu%', 'EPI', 'rate', 'Share',  \
            'RoL', 'Attach_Freq', 'Detach_Freq', \
            'Premium', 'Commission', 'Brokerage', 'Tax', 'Expected Loss', 'Reinst Prem', 'No Claims Bonus', 'Profit Commission',  'Net UW Margin', \
            'LR', 'LR after Reinst', 'MR', 'UW Ratio', 'Frequency', 'Loss Severity', 'Pure Premium before Reinst', 'Pure Premium after Reinst', \
            'Stdv Loss',  'Median Loss', \
            #'10Y Loss', '50Y Loss', '100Y Loss', '200Y Loss', '250Y Loss', \
            '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '200Y SF Loss', '250Y SF Loss', \
            'SA SF Loss', 'Ctrb SF Loss', 'Div SF Loss', \
            'Stdv LR',  '10Y SF LR', '50Y SF LR', '100Y SF LR',  '200Y SF LR', '250Y SF LR', \
            'SA SF LR', 'Ctrb SF LR', 'Div SF LR', 'Hit rate', 'Attach Return Period', 'Loss Return Period', \
            'Stdv Net Margin', '10Y SA Capital', '50Y SA Capital', '100Y SA Capital', '200Y SA Capital', '250Y SA Capital', \
            'SA Capital', 'Ctrb Capital', 'Marg Stdv Net Margin', 'Div Capital', \
            'SA Sharpe Ratio', '50Y SA RoC', '100Y SA RoC', \
            'SA RoC', 'Ctrb RoC', 'Marg Sharpe Ratio', 'Div RoC', \
            'Prob of Neg Margin']
    # display table for struct econ stats
    def set_column_setting_for_struct_econ_stats():
        col_order_dict =      ['Type', 'Limit', 'rate',       'RoL', 'Hit rate',    'LR',   'MR',  'UW Ratio',   'Prob of Neg Margin', 'Premium', 'Deductions', 'Expected Loss', 'Reinst/PC/NCB',  'Net UW Margin', 'Stdv Net Margin', 'SA Capital', 'Ctrb Capital', 'Marg Stdv Net Margin', 'Div Capital' ]
        col_string_converter = ['%s',  float_sep, perc_high,  perc_med, perc_med, perc_low, perc_low, perc_low,     perc_low,            float_sep ,  float_sep,     float_sep,   float_sep,       float_sep,         float_sep,         float_sep,       float_sep,        float_sep,        float_sep]
        col_width =            [ 70,     90,      60,          60,          75,       60,      60,      85,              80,                100,         100,           105,            100,                110,                95,            100,           105,               100  ,           100  ]
        #col_order_dict =      ['Type', 'Limit', 'rate',       'RoL', 'Hit rate',    'LR',   'MR',  'UW Ratio',  'SA Sharpe Ratio',    'SA RoC',  'Ctrb RoC',   'Marg Sharpe Ratio',   'Div RoC', 'Prob of Neg Margin', 'Premium', 'Deductions', 'Expected Loss', 'Reinst/PC/NCB',  'Net UW Margin', 'Stdv Net Margin', 'SA Capital', 'Ctrb Capital', 'Marg Stdv Net Margin', 'Div Capital' ]
        #col_string_converter = ['%s',  float_sep, perc_high,  perc_med, perc_med, perc_low, perc_low, perc_low,    '%.2f',           perc_low,    perc_low,     '%.2f',              perc_low,    perc_low,            float_sep ,  float_sep,     float_sep,   float_sep,       float_sep,         float_sep,         float_sep,       float_sep,        float_sep,        float_sep]
        #col_width =            [ 50,     90,      60,          60,          60,       50,      50,      50,         60,                60,         60,            60,                 60,          60,                80,         75,            80,             75,                75,                75,            75,                80,                75  ,           80  ]
        return col_order_dict, col_string_converter, col_width
    settings_d['olv_setting_function'] = set_column_setting_for_struct_econ_stats
    inoutput_d[name] = settings_d
    
    name = 'Prog_Economics'
    settings_d = {}
    settings_d['show'] = True
    col_order_header = ['Contract_ID', 'Layer_ID', 'Legal Carrier', 'Cedent', 'Program', 'PType', 'CType', 'LOB', 'UW Year', 'Type', 'ORIG_CCY', 'Share']
    col_order_values = ['Limit', 'Premium', 'Commission', 'Brokerage', 'Tax', 'Expected Loss', 'Reinst Prem', 'No Claims Bonus', 'Profit Commission', 'Net UW Margin']
    col_order_values_ext = list(col_order_values)
    col_order_values_ext.extend(['Stdv Loss', 'Median Loss', 'SA SF Loss', \
                'Ctrb SF Loss', 'Div SF Loss', 'Stdv Net Margin', 'SA Capital', 'Ctrb Capital', 'Marg Stdv Net Margin', 'Div Capital'])
    col_order_stats = ['LR', 'MR', 'UW Ratio', 'Stdv LR', 'SA SF LR', 'Ctrb SF LR', 'Div SF LR', 'Hit rate', \
                    'SA Sharpe Ratio', 'SA RoC', 'Ctrb RoC', 'Marg Sharpe Ratio', 'Div RoC', 'Prob of Neg Margin']
    settings_d['col_order_header'] = col_order_header
    settings_d['col_order_values'] = col_order_values
    settings_d['col_order_values_ext'] = col_order_values_ext
    settings_d['col_order_stats'] = col_order_stats
    col_order = list(col_order_header)
    col_order.extend(col_order_values)
    settings_d['col_order'] = col_order
    col_order_ext = list(col_order_header)
    col_order_ext.extend(col_order_values_ext)
    col_order_ext.extend(col_order_stats)
    settings_d['col_order_ext'] = col_order_ext
    inoutput_d[name] = settings_d
    
    name = 'Prog_Exposures'
    settings_d = {}
    settings_d['show'] = True
    # output only
    if config_d['limit_per_name_flag']:
        settings_d['col_order'] = ['Contract_ID', 'Layer_ID', 'Legal Carrier', 'Cedent', 'Program', 'PType', 'CType', 'LOB', 'UW Year', 'Structure', 'Type', 'ORIG_CCY', 'Share', 'Company Name', 'Country', 'Cardinality', 'Rating', 'PD', 'R', 'Limit', 'Notional', 'PML_Notl', 'PEL_Notl', 'Limit_ORIG_CCY_100', 'Notional_ORIG_CCY_100', 'PML_Notl_ORIG_CCY_100', 'PEL_Notl_ORIG_CCY_100', 'EL_unlimited', 'ORIG_CP_NAME']
    else:
        settings_d['col_order'] = ['Contract_ID', 'Layer_ID', 'Legal Carrier', 'Cedent', 'Program', 'PType', 'CType', 'LOB', 'UW Year', 'Structure', 'Type', 'ORIG_CCY', 'Share', 'Company Name', 'Country', 'Cardinality', 'Rating', 'PD', 'R', 'Notional', 'PML_Notl', 'PEL_Notl', 'Notional_ORIG_CCY_100', 'PML_Notl_ORIG_CCY_100', 'PEL_Notl_ORIG_CCY_100', 'EL_unlimited', 'ORIG_CP_NAME']
    col_width_d = {}
    col_width_d['A:C'] = 10
    col_width_d['D:E'] = 15
    col_width_d['N:N'] = 25
    if config_d['limit_per_name_flag']:
        col_width_d['T:AB'] = ( 15, 'fmt_numbers')
    else:
        col_width_d['T:Z'] = ( 15, 'fmt_numbers')
    settings_d['col_width_d'] = col_width_d
    inoutput_d[name] = settings_d
    
    name = 'UW_Account'
    settings_d = {}
    settings_d['show'] = False
    inoutput_d[name] = settings_d
    
    name = 'config'
    settings_d = {}
    settings_d['show'] = False
    inoutput_d[name] = settings_d
    
    # override config setting with program_df info for actual run
    config_d['xl_on_qs_ret_flag'] = bool(prog_df['XL on QS Ret'].max())
    config_d['exp_struct_flag'] = bool(prog_df['Exp Struct'].max())
    config_d['group_by_cp_flag'] = bool(prog_df['Group by CP'].max())
    config_d['severity_by_policy_flag'] = bool(prog_df['Severity by Policy'].max())
    config_d['uw_year_split_flag'] = int(prog_df['UW Year Split'].max())
    config_d['run_off_flag'] = bool(prog_df['Exp Run Off'].max())
    config_d['named_bin_flag'] = bool(prog_df['Auto Bands'].max())
            
    # adjust to client specific output parameters
    if config_d['client_spec_flag'] == 1:         
        adjust_inoutput_dict_for_jlt(inoutput_d, config_d)
    else:
        # all other setting None
        pass
            
    return inoutput_d


def adjust_inoutput_dict_for_jlt(inoutput_d, config_d):
    
    def float_sep(val):
        if np.isnan(val):
            return ''
        else:
            return '{:,.0f}'.format(val)
    def float_dec(val):
        if np.isnan(val):
            return ''
        else:
            return '{:,.2f}'.format(val) 
    def perc_high(val):
        if np.isnan(val):
            return ''
        else:
            return '{:.2%}'.format(val)
    def perc_med(val):
        if np.isnan(val):
            return ''
        else:
            return '{:.1%}'.format(val)
    def perc_low(val):
        if np.isnan(val):
            return ''
        else:
            return '{:.0%}'.format(val) 
    
    name = 'Program'
    settings_d = inoutput_d[name]
    # formatting ouput
    col_order = ['Cedent', 'Program', 'Type', 'LOB', 'UW Year', 'Country', 'Currency', 'ExRate', 'Def Rating', 'Growth factor', 'Expense Ratio']
    if config_d['exp_struct_flag']:
        col_order.extend(['Exp Struct'])
    if config_d['xl_on_qs_ret_flag']:
        col_order.extend(['XL on QS Ret'])
    if config_d['group_by_cp_flag']:
        col_order.extend(['Group by CP'])
    if config_d['severity_by_policy_flag']:
        col_order.extend(['Severity by Policy'])
    if config_d['run_off_flag']:
        col_order.extend(['Exp Run Off'])
    if config_d['named_bin_flag']:
        col_order.extend(['Auto Bands'])
    settings_d['col_order'] = col_order
    col_width_d = {}
    col_width_d['A:B'] = 20
    col_width_d['F:F'] = 15
    col_width_d['I:Q'] = 12
    settings_d['col_width_d'] = col_width_d
    
    
    name = 'Portfolio'
    settings_d = inoutput_d[name]
    # formatting output (includes Risk Factor)
    col_order = ['Company Name', 'Country', 'LOB', 'Rating', 'PD', 'R', 'Cardinality', 'LNL', 'Notional', 'Notional Ceded', 'Notional Retained', 'Tenor', 'PEL', 'PML', 'Shape', 'PML_90th']
    if config_d['mf_flag']:
        col_order.extend(['RiskFactor'])
    if config_d['exp_struct_flag']:
        col_order.extend(['Attach', 'Detach'])
    settings_d['col_order'] = col_order
    inoutput_d[name] = settings_d
    
    
    name = 'Structure'
    settings_d = inoutput_d[name]
    # foramtting output
    col_order = ['Type']
    if config_d['xl_on_qs_ret_flag'] or config_d['cession_flag']:
        col_order.extend(['Cession'])
    col_order.extend(['Share', 'Attach', 'Limit', 'AggAttach', 'AggLimit', 'EPI', 'rate', 'Commission', 'Brokerage', 'Tax', 'No Claims Bonus', 'Profit Comm', 'Mgmt Exp', 'ReinstPrem'])
    if config_d['reinst_flag']:
        col_order.extend(['ReinstPrem2', 'ReinstPrem3', 'ReinstPrem4', 'ReinstPrem5'])
    col_order.extend(['Num Reinst', 'Interlocking'])
    if config_d['slide_flag']:
        col_order.extend(['SlidingScale', 'SS_ID'])
    settings_d['col_order'] = col_order
    col_width_d = {}
    if config_d['xl_on_qs_ret_flag'] or config_d['cession_flag']:
        col_width_d['C:D'] = ( 8, 'fmt_perc')
        col_width_d['E:I'] = ( 15, 'fmt_numbers')
        col_width_d['J:J'] = ( 8, 'fmt_rate')
        col_width_d['K:U'] = ( 8, 'fmt_perc')
    else:
        col_width_d['C:C'] = ( 8, 'fmt_perc')
        col_width_d['D:H'] = ( 15, 'fmt_numbers')
        col_width_d['I:I'] = ( 8, 'fmt_rate')
        col_width_d['J:T'] = ( 8, 'fmt_perc')
    settings_d['col_width_d'] = col_width_d
    
    name = 'Ptf_Stats'
    settings_d = inoutput_d[name]
    # output only
    col_order_ext = list(settings_d['col_order'])
    col_order_ext.extend(['Expected Loss', 'Stdv Loss', 'Median Loss', '10Y Loss', '50Y Loss',  '100Y Loss', '200Y Loss', '250Y Loss', '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '200Y SF Loss', '250Y SF Loss', 'Div SF Loss', 'Stdv LR', '10Y SF LR','50Y SF LR',  '100Y SF LR', '200Y SF LR', '250Y SF LR', 'Div SF LR', 'Hit rate', 'Frequency', 'Loss Severity'])
    settings_d ['col_order_ext'] = col_order_ext
    
    name = 'Exposure Stats'
    settings_d = inoutput_d[name]
    # display table for struct exposure stats
    def set_column_settings_for_struct_exp_stats():
        col_order_dict =      [ 'Type', 'Limit', 'rate', '# Notl',    'Notl%', '# PML',  'PEL_Notl%', 'EL_gu%',   'EPI',    'EL_unlimited',   'RoL', 'Attach_Freq', 'Detach_Freq']
        col_string_converter = ['%s', float_sep, perc_high, float_sep, perc_med, float_sep, perc_med, perc_high,  float_sep,    float_sep,  perc_med, float_dec,  float_dec]
        col_width =            [ 40,     85,      65,      75,          70,       75,        85,        85,        100,        110,            65,     100,         100]
        #col_order_dict =      [ 'Type', 'Limit', '# Notl', 'Notl%', '# PML',  'PEL_Notl%', 'EL_gu%',  'EL_unlimited',  'EPI',     'rate',  'RoL', 'Prem_on_Notl', 'Prem_on_PEL', 'Prem_Multiple']
        #col_string_converter = ['%s', float_sep, float_sep, perc_med, float_sep, perc_med, perc_high,  float_sep,    float_sep, perc_high,  perc_med, perc_high,  perc_high,     float_dec]
        #col_width =            [ 50,     90,        60,      60,       60,        65,      60,        80,             80,         60,       60,         85,           85,           80]
        return col_order_dict, col_string_converter, col_width 
    settings_d['olv_setting_function'] = set_column_settings_for_struct_exp_stats

    name = 'Ground-up Stats'
    settings_d = inoutput_d[name]
    # display table for ground up portfolio stats
    def set_column_settings_for_gup_stats():
        col_order_dict =      [ 'LR',    'Stdv LR',  '10Y SF LR', '50Y SF LR', '100Y SF LR',  '250Y SF LR'  , 'Hit rate', 'Premium', 'Expected Loss', 'Stdv Loss', 'Median Loss', '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '250Y SF Loss']
        col_string_converter = [perc_low, perc_low,  perc_low,    perc_low,    perc_low,     perc_low,      perc_med,     float_sep,    float_sep,      float_sep,  float_sep,      float_sep,      float_sep,     float_sep,     float_sep]
        col_width =            [ 60,     80,        90,            90,             90,       90,              80,             100,       105,               90,          95,         95,           95,              100,         100]
        #col_order_dict =      [ 'LR',    'Stdv LR', '10Y SF LR', '50Y SF LR', '100Y SF LR', '250Y SF LR',  'Hit rate', 'Premium', 'Expected Loss', 'Stdv Loss', 'Median Loss', '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '250Y SF Loss']
        #col_string_converter = [perc_low, perc_low,  perc_low,    perc_low,    perc_low,   perc_low,      perc_med, float_sep,    float_sep,      float_sep,  float_sep,      float_sep,    float_sep,     float_sep,       float_sep]
        #col_width =            [ 60,     60,        70,             70,        70,           70,           65,              90,       90,               80,          80,         80,          90,            90,             90]
        return col_order_dict, col_string_converter, col_width
    settings_d['olv_setting_function'] = set_column_settings_for_gup_stats
    
    name = 'Layer_Stats'
    settings_d = inoutput_d[name]
    # output only
    settings_d['col_order'] = ['Layer_ID', 'Type', 'Attach', 'Limit', 'AggAttach', 'AggLimit','# Notl', 'Notl%', '# PML', 'PEL_Notl%', 'EL_unlimited', 'EL_gu%', 'EPI', 'rate', 'Share',  \
            'Prem_on_Notl', 'Prem_on_PEL', 'Attach_Freq', 'Detach_Freq', \
            'RoL', 'Premium', 'Commission', 'Brokerage', 'Tax', 'Expected Loss', 'Reinst Prem', 'No Claims Bonus', 'Profit Commission',  'Net UW Margin', \
            'LR', 'LR after Reinst', 'MR', 'Prem_Multiple', 'Frequency', 'Loss Severity', 'Pure Premium before Reinst', 'Pure Premium after Reinst', \
            'Stdv Loss',  'Median Loss', '10Y Loss', '50Y Loss',  '100Y Loss', '200Y Loss', '250Y Loss', \
            '10Y SF Loss', '50Y SF Loss', '100Y SF Loss', '200Y SF Loss', '250Y SF Loss', 'Ctrb SF Loss', 'Div SF Loss', \
            'Stdv LR', '10Y SF LR', '50Y SF LR', '100Y SF LR', '200Y SF LR', '250Y SF LR', 'Ctrb SF LR', 'Div SF LR', 'Hit rate', 'Attach Return Period', 'Loss Return Period', \
            'Stdv Net Margin', '10Y SA Capital', '50Y SA Capital', '100Y SA Capital', '200Y SA Capital', '250Y SA Capital', 'Ctrb Capital', 'Marg Stdv Net Margin', 'Div Capital', \
            'SA Sharpe Ratio', '50Y SA RoC', '100Y SA RoC', 'Ctrb RoC', 'Marg Sharpe Ratio', 'Div RoC', \
            'Prob of Neg Margin']
    # display table for struct econ stats
    def set_column_setting_for_struct_econ_stats():
        col_order_dict =      ['Type', 'Limit', 'rate',       'RoL', 'Hit rate',    'LR',   'MR',  'Prem_Multiple',   'SA Sharpe Ratio',    '50Y SA RoC',  '100Y SA RoC',    'Prob of Neg Margin', 'Premium', 'Deductions', 'Expected Loss', 'Reinst/PC/NCB',  'Net UW Margin', 'Stdv Net Margin', '10Y SA Capital', '50Y SA Capital', '100Y SA Capital', '250Y SA Capital',  'Ctrb Capital' ]
        col_string_converter = ['%s',  float_sep, perc_high,  perc_med, perc_med, perc_low, perc_low, float_dec,     float_dec,           perc_low,    perc_low,              perc_low,            float_sep ,  float_sep,     float_sep,   float_sep,       float_sep,         float_sep,         float_sep,       float_sep,        float_sep,        float_sep,             float_sep]
        col_width =            [ 70,     90,      60,          60,          75,       60,      60,      105,              100,                100,         100,                    100,                100,         100,           115,            110,                120,                105,            110,           105,               105  ,           110,                   105  ]
        #col_order_dict =      ['Type', 'Limit', 'rate',       'RoL', 'Hit rate',    'LR',   'MR',  'Prem_Multiple',  'SA Sharpe Ratio',    '50Y SA RoC',  '100Y SA RoC',  'Prob of Neg Margin', 'Premium', 'Deductions', 'Expected Loss', 'Reinst/PC/NCB',  'Net UW Margin', 'Stdv Net Margin', '10Y SA Capital',  '50Y SA Capital',  '100Y SA Capital', '250Y SA Capital', 'Ctrb Capital' ]
        #col_string_converter = ['%s',  float_sep, perc_high,  perc_med, perc_med, perc_low, perc_low,  float_dec,      float_dec,           perc_low,    perc_low,        perc_low,            float_sep ,  float_sep,     float_sep,   float_sep,       float_sep,         float_sep,         float_sep,       float_sep,        float_sep,        float_sep,           float_sep]
        #col_width =            [ 50,     90,      60,          60,          60,       50,      50,      70,           60,                60,         60,              60,                80,         75,            80,             75,                75,                75,            75,                80,                80  ,           90  ,                 80]
        return col_order_dict, col_string_converter, col_width
    settings_d['olv_setting_function'] = set_column_setting_for_struct_econ_stats
    
    name = 'Prog_Economics'
    settings_d = inoutput_d[name]
    settings_d['show'] = False
    
    
    name = 'Prog_Exposures'
    settings_d = inoutput_d[name]
    settings_d['show'] = False
    settings_d['col_order'] = ['Layer_ID', 'Cedent', 'Program', 'PType', 'CType', 'LOB', 'UW Year', 'Structure', 'Type', 'ORIG_CCY', 'Share', 'Company Name', 'Country', 'Cardinality', 'RiskFactor', 'Rating', 'PD', 'R', 'Notional', 'PML_Notl', 'PEL_Notl', 'Notional_ORIG_CCY_100', 'PML_Notl_ORIG_CCY_100', 'PEL_Notl_ORIG_CCY_100', 'EL_unlimited']
    col_width_d = {}
    col_width_d['A:B'] = 10
    col_width_d['C:D'] = 15
    col_width_d['L:L'] = 25
    col_width_d['R:X'] = ( 15, 'fmt_numbers')
    settings_d['col_width_d'] = col_width_d
    
    name = 'UW_Account'
    settings_d = inoutput_d[name]
    if config_d['gross_net_econ_flag']:
        settings_d['show'] = True
    else:
        settings_d['show'] = False
    
    name = 'CorrMatrix'
    settings_d = inoutput_d[name]
    settings_d['show'] = False
    
    name = 'config'
    settings_d = inoutput_d[name]
    settings_d['show'] = False
    
    



def main():
    os.chdir('..')
    config_d = load_config_parameters()
    print(config_d)
    
    
    
    
    
if __name__ == '__main__':
    main()