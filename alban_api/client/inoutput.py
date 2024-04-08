# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:00:15 2017

@author: alban.fauchere
"""

import os
import glob
import pandas as pd
import numpy as np
from . import configp
from . import counterparty_universe
#from . import inoutput_jlt


#  handles all import / export to Excel
name_d = {'Program': 'prog_df', 'Portfolio': 'ptf_df', 'RatingToPD': 'rating_to_pd', \
          'Structure': 'structure_df', 'SlidingScale': 'sliding_scale_df', 'CorrMatrix': 'corr_matrix_df', \
            'BKG_Ptf': 'bkg_ptf_df', \
            #'parameter': 'param_df', \
            'Country_Split': 'country_split_df', \
                'Ptf_Stats': 'ptf_econ_stats_df', 'EL_unlimited_per_CP': 'EL_unlimited_by_CP_df', 'Layer_Stats': 'result_all_sim_df', \
            'Prog_Economics': 'prog_economics_sim_df', 'Pricing_Quote': 'pricing_result_structure_sim_df', \
                'UW_Account': 'uw_account_df', \
            'Prog_Exposures': 'prog_exposure_df', 'PROG_DECOMP': 'prog_decomp_df', 'perCP_summary': 'all_econ_stats_per_cp_df'}
    

# imports all tables attached to a dictionary, and provides check_flag for tables missing
def import_tables_from_Excel(full_filename, config_d={}):
    # standard settings if config missing
    if len(config_d) == 0:
        config_d['slide_flag'] = True
        config_d['cntry_split_flag'] = False
        config_d['mf_flag'] = False
    # import
    prog_df, ptf_df, rating_to_pd, structure_df, sliding_scale_df, country_split_df, bkg_ptf_df, corr_matrix_df, param_df, check_flag = import_portfolio_raw(full_filename, config_d)
    # assign to dict
    import_d = {}
    import_d['Program'] = prog_df
    import_d['Portfolio'] = ptf_df
    import_d['RatingToPD'] = rating_to_pd
    import_d['Structure'] = structure_df
    import_d['SlidingScale'] = sliding_scale_df
    if config_d['cntry_split_flag']:
        import_d['Country_Split'] = country_split_df
    import_d['BKG_Ptf'] = bkg_ptf_df
    if config_d['mf_flag']:
        import_d['CorrMatrix'] = corr_matrix_df
    #import_d['parameter'] = param_df
    
    return import_d, check_flag


# import ptf without error handling, only checks if key tables are present, and generates empty dfs for others
def import_portfolio_raw(full_filename, config_d={}):
    # standard settings if config missing
    if len(config_d) == 0:
        config_d['slide_flag'] = True
        config_d['cntry_split_flag'] = False
        config_d['mf_flag'] = False
    
    check_flag = True
    with pd.ExcelFile(full_filename) as excelfile:
        try:
            # import program info
            prog_df = pd.read_excel(excelfile, sheet_name='Program', header=0)
            # import portfolio
            ptf_df = pd.read_excel(excelfile, sheet_name='Portfolio', header=0)
            # import rating to PD table
            rating_to_pd = pd.read_excel(excelfile, sheet_name='RatingToPD', header=0)
            # import structure
            structure_df = pd.read_excel(excelfile, sheet_name='Structure', header=0)
        except:
            print('Key tab is missing: Sheet must include Program, Portolio, Structure and RatingToPD tabs.')
            check_flag = False
            prog_df = pd.DataFrame()
            ptf_df = pd.DataFrame()
            rating_to_pd = pd.DataFrame()
            structure_df = pd.DataFrame()
        
        try: 
            # import Sliding scale (only required is slide_flag set)
            sliding_scale_df = pd.read_excel(excelfile, sheet_name='SlidingScale', header=0)
        except:
            if config_d['slide_flag']:
                print('SlidinScale tab missing.')
                check_flag = False
            sliding_scale_df = pd.DataFrame()
        
        if config_d['cntry_split_flag']:
            try:
                # import country split if available
                country_split_df = pd.read_excel(excelfile, sheet_name='Country_Split', header=0)
            except:
                country_split_df = pd.DataFrame()
        else:
            country_split_df = pd.DataFrame()
          
        try:    
            # import BKG portfolio
            bkg_ptf_df = pd.read_excel(excelfile, sheet_name='BKG_Ptf')
        except:
            print('BKG Portfolio missing')
            # potentially use client portfolio
            check_flag = False
            bkg_ptf_df = pd.DataFrame()
        
        if config_d['mf_flag']:
            try:    
                # import corr matrix
                corr_matrix_df = pd.read_excel(excelfile, sheet_name='CorrMatrix')
            except:
                print('CorrMatrix missing')
                # use pre-set
                corr_matrix_df = pd.DataFrame()
                #check_flag = False
        else:
            corr_matrix_df = pd.DataFrame()
        
        try:
            param_df = pd.read_excel(excelfile, sheet_name='parameter', header=0)
        except:
            param_df = pd.DataFrame()
            
    return prog_df, ptf_df, rating_to_pd, structure_df, sliding_scale_df, country_split_df, bkg_ptf_df, corr_matrix_df, param_df, check_flag



# clean ptf : including error handling with obligatory and optional fields, latter with default values
# distinguishes between critical error that throw a flag, from those that are corrected on the fly (e.g. min/max for parameters)
def clean_portfolio_from_settings(import_d, config_d, full_filename=None): 
    check_flag = True
    cleaned_import_d = {}
    # open error log to document detailed data corrections done
    if config_d['output_flag'] and full_filename:
        file_path = '\\'.join(full_filename.split('\\')[:-1])
        error_log = open(os.path.join(file_path, './output/log_import.txt'), 'w')
        error_log.write('Import log for ' + full_filename)
        error_log.write('\n')
    else:
        error_log = None
    
    # import setings (first without prog_df)
    inoutput_d = configp.define_inoutput_d(config_d)
    
    # check if all tables are present
    for tablename in ['Program', 'Portfolio', 'RatingToPD', 'Structure', 'SlidingScale', 'BKG_Ptf']:
        if tablename not in import_d.keys():
            check_flag=False
            break
    if 'Country_Split' not in import_d.keys() and config_d['cntry_split_flag']:
        import_d['Country_Split'] = pd.DataFrame()
    if 'CorrMatrix' not in import_d.keys() and config_d['mf_flag']:
        import_d['CorrMatrix'] = pd.DataFrame()
    if not check_flag:
        return cleaned_import_d, inoutput_d, check_flag
    
    name = 'Program'
    df = import_d[name]
    prog_df = df
    key_cols = inoutput_d[name]['key_cols']
    rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
    int_cols = inoutput_d[name]['int_cols']
    float_cols = inoutput_d[name]['float_cols']

    check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
    # check for only one row
    if len(df) !=1:
        print('Program to be defined in valid single row.')
        print()
        check_flag_new = False
    check_flag = check_flag and check_flag_new
    
    # stop in key errors for prog_df as it contains default values for other dfs
    if check_flag:
        df['Growth factor'] = df['Growth factor'].clip(0)
        if df['Growth factor'].max() == 0:
            df['Growth factor'] = 1
    else:
        return cleaned_import_d, inoutput_d, check_flag
    
    # refresh settings with prog_df
    inoutput_d = configp.define_inoutput_d(config_d, df)
    if config_d['uw_year_split_flag']:
        df['UW Year Split'] = df['UW Year Split'].clip(0,15).astype(int)
    cleaned_import_d[name] = df
    
    name = 'Structure'
    df = import_d[name]
    structure_df = df
    # check for type to be of allowed value QS or XL
    if 'Type' in df.columns:
        struct_type = df.loc[:, 'Type'].str.strip()
        qs_rows = struct_type == 'QS'
        xl_rows = struct_type == 'XL'
        allowed_rows = qs_rows | xl_rows
        if not allowed_rows.all():
            print('Structure ', struct_type[~allowed_rows].values, 'not of allowed QS or XL type eliminated.')
            df = df.loc[allowed_rows, :].copy() 
     
    key_cols = inoutput_d[name]['key_cols']
    rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
    int_cols = inoutput_d[name]['int_cols']
    float_cols = inoutput_d[name]['float_cols']
    
    check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
    
    # check that structure_df not empty (after cleaning):
    if df.empty:
        print('Structure not valid.')
        check_flag_new = False
    check_flag = check_flag and check_flag_new
    
    if check_flag_new == True:
        # dataframe column specific data restrictions
        df['Type'] = df['Type'].str.strip()        
        df['Attach'] = df['Attach'].clip(0)
        df['Limit'] = df['Limit'].clip(0)
        df['AggAttach'] = df['AggAttach'].clip(0)
        # re-set AggLimit to 1000 * EPI if missing
        df.loc[df['AggLimit'].isnull(), 'AggLimit'] = 1000 * df.loc[df['AggLimit'].isnull(), 'EPI']
        df['AggLimit'] = df['AggLimit'].clip(0)
        df['Num Reinst'] = df['AggLimit'] / df['Limit'] - 1
        df.loc[df['Type']=='QS', 'Num Reinst'] = np.nan
        df.loc[df['Type']=='QS', 'Interlocking'] = 0
        if config_d['slide_flag']:
            df['SlidingScale'] = df['SlidingScale'].apply(bool)
        if config_d['reinst_flag']:
            # set Reinst Prem to zero above num of reinst or for QS
            num_reinst = df['Num Reinst'].apply(np.ceil)
            for i in range(1,6):
                sel = (num_reinst < i) | (df['Type']=='QS')
                rname = 'ReinstPrem'
                if i>1:
                    rname += str(i)
                df.loc[sel, rname] = 0
        if prog_df['XL on QS Ret'].max() or config_d['cession_flag']:
            df['Cession'] = df['Cession'].clip(0,1)
            if not config_d['cession_flag']:
                # set cession to 1 for xl
                sel = df['Type']=='XL'
                df.loc[sel, 'Cession'] = 1.0
        df['Share'] = df['Share'].clip(0,1)
        if config_d['limit_per_name_flag']:
            df['AutoLimit'] = df['AutoLimit'].clip(0)
            # set to zero for XL
            sel = df['Type']=='XL'
            df.loc[sel, 'AutoLimit'] = 0
            
    cleaned_import_d[name] = df
    
    name = 'RatingToPD'
    df = import_d[name]
    rating_to_pd = df
    key_cols = inoutput_d[name]['key_cols']
    rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
    int_cols = inoutput_d[name]['int_cols']
    float_cols = inoutput_d[name]['float_cols']
    
    check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
    if check_flag_new == False:
        print('Rating to PD table corrupt')
    check_flag = check_flag and check_flag_new
    
    if check_flag_new == True:
        # dataframe column specific data restrictions
        df['PD'] = df['PD'].clip(0.0001, 0.9999)
    cleaned_import_d[name] = df
        
    name = 'Portfolio'
    df = import_d[name]
    key_cols = inoutput_d[name]['key_cols']
    rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
    int_cols = inoutput_d[name]['int_cols']
    float_cols = inoutput_d[name]['float_cols']
    
    check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
    if df.empty:
        print('Portfolio not valid.')
        check_flag_new = False
        
    check_flag = check_flag and check_flag_new
    if check_flag_new == True:
        # add Exp_ID
        df['Exp_ID'] = df.index
        # dataframe column specific data restrictions
        # trim name and Country string
        df['Company Name'] = df['Company Name'].str.strip()
        df['ORIG_CP_NAME'] = df['ORIG_CP_NAME'].str.strip()
        sel = df['ORIG_CP_NAME']==''
        df.loc[sel, 'ORIG_CP_NAME'] = df.loc[sel, 'Company Name']
        df['Country'] = df['Country'].str.strip()
        df['PEL'] = np.minimum(0.999, np.maximum(df['PEL'],0.001))
        df['Shape'] = np.maximum(df['Shape'],1.001)
        df['R'] = np.minimum(0.999, np.maximum(df['R'],0.0))
        # allow only half-year steps between 0.5 and 10
        max_tenor = config_d['max_tenor']
        if df['Tenor'].max() > max_tenor:
            print('Warning: Some Tenor settings were cut off at maximum allowed tenor.')
        df['Tenor'] = (df['Tenor'].clip(0.1,max_tenor)*2).apply(np.ceil)/2 
        df['LNL'] = df['LNL'].clip(0,1).astype(int)
        # card min 1
        df['Cardinality'] = df['Cardinality'].clip(1).astype(int)
        df.loc[df['Cardinality']==1, 'LNL'] = 0
        # added exposure specific attach/detach, make sure detach is larger than attach
        df['Attach'] = np.minimum(1.0, np.maximum(df['Attach'],0.0))
        df['Detach'] = np.minimum(1.0, np.maximum(df['Detach'],0.0))
        df['Detach'] = np.maximum(df['Detach'], df['Attach'])
        # make LNL=1 for large bands
        lnl_thres = config_d['lnl_thres']
        isLargeBands = ( (df['Cardinality']>lnl_thres) & (df['LNL']==0) ).any()
        if isLargeBands:
            print('LNL=1 is automatically set for bands with Cardinality > ', lnl_thres, '.')
            df.loc[df['Cardinality']>lnl_thres, 'LNL'] = 1
        # check for LNL==1 rows not to have exp-specific structure
        sel = df['LNL']==1
        isAttachDetach = (df.loc[sel, 'Attach']!=0).any() or (df.loc[sel, 'Detach']!=1).any()
        if isAttachDetach:
            print('Warning: Large N Limit (LNL) does not use distributed severity for Attach/Detach!')
            print()
        # make sure Company Name for bands has no duplicates
        # check for duplicates in risk bands
        df_unnamed = df[df['Cardinality']!=1]
        dupl_in_unnamed = df_unnamed['Company Name'].duplicated().sum()
        if dupl_in_unnamed > 0:
            print('Risk band names must be unique!')
            check_flag = False
        if config_d['limit_per_name_flag']:
            df['Limit'] = df['Limit'].clip(0)
            # enforce limit to be max on all exposures on same name
            #max_limit_lookup = df[['Company Name', 'Limit']].groupby('Company Name').max()
            #df['Limit'] = df['Company Name'].map(max_limit_lookup.to_dict()['Limit'])
        # uw year split checks
        if config_d['uw_year_split_flag']:
            # normalize non_zero pattern
            uw_year_split_flag = config_d['uw_year_split_flag']
            uy_cols = [ ('UY-'+ str(i)) for i in range(uw_year_split_flag)]
            sum_pattern_s = sum( df[uwy] for uwy in uy_cols)
            #sum_pattern_s = df['UY-0'] + df['UY-1'] + df['UY-2'] + df['UY-3'] + df['UY-4'] + df['UY-5'] + df['UY-6'] + df['UY-7'] + df['UY-8'] + df['UY-9']
            sel_zero = (sum_pattern_s == 0)
            for uwy in uy_cols:
                df.loc[~sel_zero, uwy] /= sum_pattern_s[~sel_zero]
            
        # check cp consistency of ptf_df (uniqueness of CP parameters)
        set_PDs(df, rating_to_pd)
        df, dummy = counterparty_universe.make_ptf_df_cp_consistent(df)
    cleaned_import_d[name] = df
    ptf_df = df
    
    # almost same for bkg_ptf
    name = 'BKG_Ptf'
    df = import_d[name]
    key_cols = inoutput_d[name]['key_cols']
    rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
    int_cols = inoutput_d[name]['int_cols']
    float_cols = inoutput_d[name]['float_cols']
    
    check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
    if df.empty:
        print('BKG Ptf not valid.')
        check_flag_new = False
    check_flag = check_flag and check_flag_new
    if check_flag_new == True:
        # dataframe column specific data restrictions
        # trim name and Country string
        df['Company Name'] = df['Company Name'].str.strip()
        df['PEL'] = np.minimum(0.999, np.maximum(df['PEL'],0.001))
        df['Shape'] = np.maximum(df['Shape'],1.001)
        df['R'] = np.minimum(0.999, np.maximum(df['R'],0.0))
        max_tenor = config_d['max_tenor']
        if df['Tenor'].max() > max_tenor:
            print('Warning: Some Tenor settings in BKG Ptf were cut off at maximum allowed tenor.')
        df['Tenor'] = (df['Tenor'].clip(0.1,10)*2).apply(np.ceil)/2   
        df['LNL'] = df['LNL'].clip(0,1).astype(int)
        df.loc[df['Cardinality']==1, 'LNL'] = 0
        df.loc[df['Cardinality']>lnl_thres, 'LNL'] = 1
        # check for duplicates in risk bands
        df_unnamed = df[df['Cardinality']!=1]
        dupl_in_unnamed = df_unnamed['Company Name'].duplicated().sum()
        if dupl_in_unnamed > 0:
            print('Risk band names must be unique!')
            check_flag = False
    cleaned_import_d[name] = df
    
    if config_d['mf_flag']:
        name = 'CorrMatrix'
        df = import_d[name]
        if config_d['mf_flag'] and df.empty==False:
            #print 'checking correlation matrix'
            key_cols = inoutput_d[name]['key_cols']
            rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
            int_cols = inoutput_d[name]['int_cols']
            float_cols = inoutput_d[name]['float_cols']
            
            check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log, elim_flag=False)
            mf_dim = 4
            if df.shape == (mf_dim, mf_dim) and check_flag_new:
                df.index = key_cols
                # specific checks, force diagonal and symmetry
                for i in range(4):
                    #i_index = 'RF ' + str(i)
                    df.iloc[i,i] = 1.0
                    for j in range(i, 4):
                        #j_index = 'RF ' + str(j)
                        df.iloc[j,i] = df.iloc[i,j]
            else:
                print('CorrMatrix corrupt')
                df = pd.DataFrame()
                check_flag_new = False
        # if corr matrix corrupt - no error is forced, default values are used 
        #check_flag = check_flag and check_flag_new
        cleaned_import_d[name] = df

    # now clean sliding_scale_df : require tab with proper columns if slide_flag = 1 (can be empty)
    name = 'SlidingScale'
    df = import_d[name]
    key_cols = inoutput_d[name]['key_cols']
    rel_cols_with_dfval = inoutput_d[name]['rel_cols_with_dfval']
    int_cols = inoutput_d[name]['int_cols']
    float_cols = inoutput_d[name]['float_cols']
    
    if config_d['slide_flag']:
        if df.empty:
            # make empty df with proper columns
            col_order = inoutput_d[name]['col_order']
            df = pd.DataFrame(columns=col_order)
            # check that no Sliding Scale is set to True
            sel = structure_df['SlidingScale'].any()
            if sel:
                print('SlidingScale information missing.')
                check_flag_new = False
        else:
            check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
    
            # check for at minimum one entry per SS_ID in Structure tab
            sel = structure_df['SlidingScale']==True
            req_ss_ids = structure_df.loc[sel, 'SS_ID']
            act_ss_ids = df['SS_ID'].unique()
            missing_ids = req_ss_ids[~req_ss_ids.isin(act_ss_ids)]
            if missing_ids.empty == False:
                print('SlidingScale has not entry for SS_IDs ', missing_ids.values)
                check_flag_new = False
            
        check_flag = check_flag and check_flag_new

        if check_flag_new == True:
            # dataframe column specific data restrictions
            # currently no other checks on SlidingScale values
            pass
            #df = sliding_scale_df
            #df['LossRatio'] = df['LossRatio'].clip(0)
    cleaned_import_d[name] = df
    
    if config_d['cntry_split_flag'] and check_flag:
        name = 'Country_Split'
        df = import_d[name]
        if not df.empty:
            # check proper format
            key_cols = ['Country', 'Notional']
            rel_cols_with_dfval = {'Country': 'Unknown', 'Notional': 0}
            int_cols = []
            float_cols = ['Notional']
            check_flag_new = clean_dataframe(df, name, key_cols, rel_cols_with_dfval, int_cols, float_cols, error_log)
            if check_flag_new == False:
                print('Country Split table corrupt')
            
            check_flag = check_flag and check_flag_new
            if check_flag_new:
                # dataframe column specific data restrictions
                df['Notional'] = df['Notional'].clip(0)
                df['Allocation'] = df['Notional'] / df['Notional'].sum()
        else:
            # generate country split from ptf_df
            xl_on_qs_ret_flag = bool(prog_df['XL on QS Ret'].max())
            df = determine_country_allocation_from_ptf_df(ptf_df, xl_on_qs_ret_flag)
        # only attached if enabled
        cleaned_import_d[name] = df
    
    # param_df not yet checked
    #cleaned_import_d['parameter'] = import_d['parameter']
    
    if error_log:
        error_log.close()
        
    return cleaned_import_d, inoutput_d, check_flag



# determine per country split for total ptf
def determine_country_allocation_from_ptf_df(ptf_df, xl_on_qs_ret_flag=True):
    country_split_df = pd.DataFrame()
    ptf_df_copy = ptf_df.copy()
    if not xl_on_qs_ret_flag:
        ptf_df_copy['Notional'] = ptf_df_copy['Notional Retained'] + ptf_df_copy['Notional Ceded']
    ptf_df_copy['Total Notl'] = ptf_df_copy['Notional']*ptf_df_copy['Cardinality']
    tot_notl = ptf_df_copy['Total Notl'].sum()
    exp_cntry_tot_s = ptf_df_copy.groupby('Country')['Total Notl'].sum()
    
    country_split_df['Notional'] = exp_cntry_tot_s
    country_split_df['Allocation'] = exp_cntry_tot_s / tot_notl
    country_split_df.reset_index(inplace=True)
    country_split_df.sort_values(by='Notional', ascending=False, inplace=True)
        
    return country_split_df


# lookup PD by rating, missing are set to 1%
def set_PDs(ptf_df, rating_to_pd):
    rat_pd_dict = rating_to_pd.set_index('Rating').to_dict()
    rat_pd_dict = rat_pd_dict['PD']
    ptf_df['PD'] = ptf_df['Rating'].map(rat_pd_dict)
    if ptf_df['PD'].isnull().any():
        print('Ratings invalid for ', ptf_df.loc[ptf_df['PD'].isnull(), 'Company Name'])
        print('Set default PD to 0.01')
        print()
    ptf_df['PD'].fillna(0.01, inplace=True)
    
    
# generic cleaning function (inplace)
def clean_dataframe(df, name, key_cols, relevant_cols_to_def_dict, int_cols, float_cols, error_log=None, drop_flag=True, elim_flag=True):
    
    # check for key columns
    if pd.Series(key_cols).isin(df.columns).all() == False:
        print(name, ': Key colums are missing.')
        print('Obligatory colums are: ', key_cols)
        return False
  
    # eliminate non-relevant columns
    cols_with_def_val = list(relevant_cols_to_def_dict.keys())
    relevant_cols = set(key_cols + cols_with_def_val)
    other_cols = list(df.columns[~df.columns.isin(relevant_cols)])
    if drop_flag:
        df.drop(other_cols, axis=1, inplace=True)
    
    # eliminate rows with nan entry in key columns without default value
    key_cols_s = pd.Series(key_cols)
    key_cols_wo_def = list(key_cols_s[~key_cols_s.isin(cols_with_def_val)])
    rows_with_key_col_error = df[key_cols_wo_def].isnull().any(axis=1)
    if rows_with_key_col_error.any() == True and elim_flag:
        num_errors = rows_with_key_col_error.sum()
        print(name, ': Eliminated ', num_errors, 'rows with missing key fields - check import_log')
        if error_log != None:
            error_log.write((str(name) +': Eliminated ' + str(num_errors) + ' rows with missing key fields'))
            error_log.write('\n')
            try:
                error_log.write(df.loc[rows_with_key_col_error, key_cols].to_string())
            except:
                error_log.write(str(df.loc[rows_with_key_col_error, key_cols].index))
            error_log.write('\n')
        # debug only
        #print df.loc[rows_with_key_col_error, key_cols_wo_def
        df.drop(df.index[rows_with_key_col_error], inplace=True)
    # alternative: df.dropna(subset=key_cols_wo_def, inplace=True)

    # add missing columns
    for col in relevant_cols_to_def_dict:
        if col in df.columns:
            pass
        else:
            df[col] = relevant_cols_to_def_dict[col]
        
    # convert to expected dtype by columns (with fillna done after coercion as it may produce nan)
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce')
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
    
    # fill NA values in non-key columns
    df.fillna(value=relevant_cols_to_def_dict, inplace=True)       

    # convert to proper type (Note: nan does not convert to int)
    df[int_cols] = df[int_cols].astype(int) 
    df[float_cols] = df[float_cols].astype(float)

    return True
 


# writing output to a new file, only filling tabs where data is not empty
# includes writing out prog_decomp_df
def export_results_to_Excel_raw(result_filename, df_dict):
    
    fmt_numbers = {'align': 'right', 'num_format': '#,##0'} 
    fmt_decimal = {'align': 'right', 'num_format': '0.00'} 
    fmt_perc = {'align': 'right', 'num_format': '0%'}
    fmt_rate = {'align': 'right', 'num_format': '0.00%'}
    format_d = {'fmt_numbers': fmt_numbers, 'fmt_decimal': fmt_decimal, 'fmt_perc': fmt_perc, 'fmt_rate': fmt_rate}
    
    writer = pd.ExcelWriter(result_filename, engine='xlsxwriter')
    wb = writer.book
    
    name = 'Program'
    if name in df_dict.keys():
        df = df_dict[name]
        df.to_excel(writer, sheet_name=name, index=False)
    
    name = 'Portfolio'
    if name in df_dict.keys():
        df = df_dict[name]
        df.to_excel(writer, sheet_name=name, index=False)
    
    name='Country_Split'
    if name in df_dict.keys() and not df.empty:
        df = df_dict[name]
        df.to_excel(writer, sheet_name=name, index=False)
    
    name = 'Structure'
    if name in df_dict.keys():
        structure_df = df_dict[name]
        structure_df.to_excel(writer, sheet_name=name)
    
    name = 'SlidingScale'
    if name in df_dict.keys() and not df.empty:
        df = df_dict[name]
        df.to_excel(writer, sheet_name=name, index=False)
    
    name = 'Ptf_Stats'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        df['Expected Loss'] *= -1
        ptf_rename = {'QS': 'Ceded to QS', 'XL': 'Retained to XL', 'TOT': 'Total Ptf'}
        df.rename(index=ptf_rename, inplace=True)
        df.rename(columns={'EL': 'EL_analytic'}, inplace=True)
        df.T.to_excel(writer, sheet_name=name)
        #ws = writer.sheets[name]
        #apply_formats(wb, ws, df.T)    
    
    name= 'EL_unlimited_per_CP'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            df.rename(columns=struct_rename, inplace=True)
            df.to_excel(writer, sheet_name=name, index=True)
    
    name = 'Layer_Stats'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df['Prem_Multiple'] = 1 / df['LR']
            struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            df.rename(index=struct_rename, inplace=True)
            df.T.to_excel(writer, sheet_name=name)
            #ws = writer.sheets[name]
            #apply_formats(wb, ws, df.T)
    
    name = 'Pricing_Quote'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df['Prem_Multiple'] = 1 / df['LR']
            struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            df.rename(index=struct_rename, inplace=True)
            df.T.to_excel(writer, sheet_name=name)
            #ws = writer.sheets['Pricing_Quote']
            #apply_formats(wb, ws, df.T)
        
    name= 'Prog_Economics'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df.T.to_excel(writer, sheet_name=name)
            #ws = writer.sheets[name]
            #apply_formats(wb, ws, prog_economics_df.T)
    
    name = 'Prog_Exposures'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name=name, index=False)
    
    name = 'UW_Account'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name=name, index=True)
            #ws = writer.sheets[name]
            #df.reset_index('Type',inplace=True)
            #apply_formats(wb, ws, df, uw_account_flag=True)
    
    name = 'PROG_DECOMP'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name='PROG_DECOMP')
            
    name = 'perCP_summary'
    if name in df_dict.keys():
        df = df_dict[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name=name, index=False)
            #ws = writer.sheets[name]
        
    writer.save()
    
    
    

# exports result tables to Excel, config/inoutput determining format
def export_results_to_Excel(result_filename, result_d={}, config_d={}, inoutput_d={}):
    # config_d and Program required, inoutput if missing generated from system
    if len(inoutput_d)==0:
        try:
            prog_df = result_d['Program']
        except:
            prog_df = pd.DataFrame()
        inoutput_d = configp.define_inoutput_d(config_d, prog_df)
        #print(inoutput_d)
    
    # available tables
    table_list = result_d.keys()
    # assign to df name
    for tablename in result_d.keys():
        if tablename in name_d:
            df_name = name_d[tablename]
            locals()[df_name] = result_d[tablename]
            
    fmt_numbers = {'align': 'right', 'num_format': '#,##0'} 
    fmt_decimal = {'align': 'right', 'num_format': '0.00'} 
    fmt_perc = {'align': 'right', 'num_format': '0%'}
    fmt_rate = {'align': 'right', 'num_format': '0.00%'}
    format_d = {'fmt_numbers': fmt_numbers, 'fmt_decimal': fmt_decimal, 'fmt_perc': fmt_perc, 'fmt_rate': fmt_rate}
    
    # write to Excel
    writer = pd.ExcelWriter(result_filename, engine='xlsxwriter')
    wb = writer.book
    
    
    name = 'Program'
    if name in table_list and name in name_d:
        df = result_d[name].copy()
        col_order = inoutput_d[name]['col_order']
        col_width_d = inoutput_d[name]['col_width_d']
        df = df[col_order]
        df.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        for col in col_width_d:
            tup = col_width_d[col]
            if type(tup) == tuple:
                fmt_obj = wb.add_format(format_d[col_width_d[col][1]])
                ws.set_column(col, col_width_d[col][0], fmt_obj)
            else:
                ws.set_column(col, col_width_d[col])
    
    
    name = 'Portfolio'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        col_order = inoutput_d[name]['col_order']
        col_width_d = inoutput_d[name]['col_width_d']
        df = df[col_order]
        df.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        for col in col_width_d:
            tup = col_width_d[col]
            if type(tup) == tuple:
                fmt_obj = wb.add_format(format_d[col_width_d[col][1]])
                ws.set_column(col, col_width_d[col][0], fmt_obj)
            else:
                ws.set_column(col, col_width_d[col])
    
    name = 'Country_Split'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name='Country_Split', index=False)
            ws = writer.sheets['Country_Split']
            ws.set_column('A:A', 20)
            fmt_obj = wb.add_format(fmt_numbers)
            ws.set_column('B:B', 15, cell_format=fmt_obj)
            fmt_obj = wb.add_format(fmt_perc)
            ws.set_column('C:C', 10, cell_format=fmt_obj)
        
    name = 'Structure'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        structure_df = df.copy()
        col_order = inoutput_d[name]['col_order']
        col_width_d = inoutput_d[name]['col_width_d']
        df = df[col_order]    
        df.to_excel(writer, sheet_name=name)
        ws = writer.sheets[name]
        for col in col_width_d:
            tup = col_width_d[col]
            if type(tup) == tuple:
                fmt_obj = wb.add_format(format_d[col_width_d[col][1]])
                ws.set_column(col, col_width_d[col][0], fmt_obj)
            else:
                ws.set_column(col, col_width_d[col])
    
    name = 'SlidingScale'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        col_order = inoutput_d[name]['col_order']
        #if not df.empty:
        df = df[col_order]
        col_width_d = inoutput_d[name]['col_width_d']
        df.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        for col in col_width_d:
            tup = col_width_d[col]
            if type(tup) == tuple:
                fmt_obj = wb.add_format(format_d[col_width_d[col][1]])
                ws.set_column(col, col_width_d[col][0], fmt_obj)
            else:
                ws.set_column(col, col_width_d[col])

    name = 'Ptf_Stats'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        ptf_rename = inoutput_d[name]['ptf_rename']
        # check if stats are only analytical or include simulation stats    
        if 'Stdv Loss' in df.columns:
            col_order = inoutput_d[name]['col_order_ext']
            df['Expected Loss'] *= -1
        else:
            col_order = inoutput_d[name]['col_order']
        df.rename(index=ptf_rename, inplace=True)
        df = df[col_order]
        df.rename(columns={'EL': 'EL_analytic'}, inplace=True)
        df.T.to_excel(writer, sheet_name=name)
        ws = writer.sheets[name]
        apply_formats(wb, ws, df.T)    
    
    
    name= 'EL_unlimited_per_CP'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty and inoutput_d[name]['show']:
            struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            df.rename(columns=struct_rename, inplace=True)
            df.to_excel(writer, sheet_name=name, index=True)
            ws = writer.sheets[name]
            #fmt_alpha = {'align': 'left'}
            ws.set_column('A:A', 25)  #, fmt_alpha)
            #ws.set_column('B:D', 10, fmt_alpha)
            fmt_numbers = {'align': 'right', 'num_format': '#,##0'}
            fmt_obj = wb.add_format(fmt_numbers)
            ws.set_column('E:M', 12, fmt_obj)
    
    
    name = 'Layer_Stats'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty:
            col_order = inoutput_d[name]['col_order']
            #df.rename(columns={'EL': 'EL_unlimited'}, inplace=True)
            df['Prem_Multiple'] = 1 / df['LR']
            df = df[col_order]
            struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            df.rename(index=struct_rename, inplace=True)
            df.T.to_excel(writer, sheet_name=name)
            ws = writer.sheets[name]
            apply_formats(wb, ws, df.T)
    
    name = 'Pricing_Quote'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty:
            name_format = 'Layer_Stats'
            col_order = inoutput_d[name_format]['col_order']
            #df.rename(columns={'EL': 'EL_unlimited'}, inplace=True)
            df['Prem_Multiple'] = 1 / df['LR']
            df = df[col_order]
            struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            df.rename(index=struct_rename, inplace=True)
            df.T.to_excel(writer, sheet_name=name)
            ws = writer.sheets[name]
            apply_formats(wb, ws, df.T)
        
    name= 'Prog_Economics'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty and inoutput_d[name]['show']:
            #struct_rename = pd.Series((structure_df.index.astype(str) + '_' + structure_df['Type']), index=structure_df.index)
            #df.rename(index=struct_rename, inplace=True)
            df.T.to_excel(writer, sheet_name=name)
            ws = writer.sheets[name]
            apply_formats(wb, ws, df.T)
    
    # exposure calc
    name = 'Prog_Exposures'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty and inoutput_d[name]['show']:
            col_order = inoutput_d[name]['col_order']
            df = df[col_order]
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            col_width_d = inoutput_d[name]['col_width_d']
            for col in col_width_d:
                tup = col_width_d[col]
                if type(tup) == tuple:
                    fmt_obj = wb.add_format(format_d[col_width_d[col][1]])
                    ws.set_column(col, col_width_d[col][0], fmt_obj)
                else:
                    ws.set_column(col, col_width_d[col])
    
    name = 'UW_Account'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty and inoutput_d[name]['show']:
            df.to_excel(writer, sheet_name=name, index=True)
            ws = writer.sheets[name]
            # data from json w/o index
            df.set_index('Category', inplace=True)
            #df.reset_index('Type',inplace=True)
            apply_formats(wb, ws, df, uw_account_flag=True)
            ws.set_column('A:B', 30)
            ws.set_column('C:N', 13)
    
    name = 'PROG_DECOMP'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name=name)
            ws = writer.sheets[name]
    
    name = 'perCP_summary'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if not df.empty:
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            #apply_formats(wb, ws, df)
            
    name = 'CorrMatrix'
    if name in table_list and name in name_d:
        df= result_d[name].copy()
        if config_d['mf_flag'] and inoutput_d[name]['show']:
            df.to_excel(writer, sheet_name=name)
    
    name = 'config'
    if config_d['debug_flag'] and inoutput_d[name]['show']:
        config_s = pd.Series(config_d)
        config_df = pd.DataFrame(config_s)
        config_df.to_excel(writer, sheet_name=name)
        ws = writer.sheets[name]
        ws.set_column('A:B', 20)

    writer.save()
    #writer.close()
    


def apply_formats(wb, ws, df, uw_account_flag=False):
    
    fmt_numbers = {'align': 'right', 'num_format': '#,##0'} 
    fmt_decimal = {'align': 'right', 'num_format': '0.00'} 
    fmt_perc = {'align': 'right', 'num_format': '0%'}
    fmt_rate = {'align': 'right', 'num_format': '0.00%'}
    
    for i, row_name in enumerate(df.index):
            fmt = None
            if row_name in ('Notional', 'Avg_Notl', 'Max_Notl', 'EPI', 'EL', 'EL_analytic', 'EL_unlimited', 'Attach', 'Limit', 'AggAttach', 'AggLimit', \
            'Premium', 'Commission', 'Brokerage', 'Tax', 'Expected Loss', 'Reinst Prem', 'Margin before PC', 'No Claims Bonus', 'Profit Commission', 'Net UW Margin', \
            'Stdv Loss', 'Median Loss', 'SA SF Loss', 'Ctrb SF Loss', 'Div SF Loss', \
            'Stdv Net Margin', 'SA Capital',  'Ctrb Capital', 'Marg Stdv Net Margin', 'Div Capital', \
            'Claims', 'Expenses', 'Result', 'Margin - Standard Deviation' ):
                fmt = fmt_numbers
            elif row_name in ('PEL', 'PML', 'LR', 'LR_Named', 'MR', 'UW Ratio', 'LR after Reinst', r'Notl_Named%', r'Notl%', r'PML_Notl%', r'PEL_Notl%', 'SA RoC', '50Y SA RoC', '100Y SA RoC', 'Ctrb RoC', 'Div RoC'):
                fmt = fmt_perc
            elif row_name in ('SA Sharpe Ratio', 'Marg Sharpe Ratio', 'Prem_Multiple', 'Attach_Freq', 'Detach_Freq', 'Frequency', 'Attach Return Period', 'Loss Return Period'):
                fmt = fmt_decimal
            elif row_name in ('PD', 'Prem_Rate', 'Share', 'ELonEPI', 'RoL', 'rate', 'EL_gu%',  'Pure Premium before Reinst', 'Pure Premium after Reinst', 'Hit rate', 'Prob of Neg Margin', 'Prem_on_Notl', 'Prem_on_PEL', 'Probability of Negative Margin'):
                fmt = fmt_rate
            elif 'Sharpe' in row_name:
                fmt = fmt_decimal
            elif 'Ratio' in row_name:
                fmt = fmt_perc
            elif 'Return' in row_name:
                fmt = fmt_rate
            elif 'Loss' in row_name:
                fmt = fmt_numbers
            elif 'Capital' in row_name:
                fmt = fmt_numbers
            elif 'Deviation' in row_name:
                fmt = fmt_numbers
            elif '_perc' in row_name:
                fmt = fmt_perc
            elif 'LR' in row_name:
                fmt = fmt_perc
                
            if fmt:
                fmt_obj = wb.add_format(fmt)
                ws.set_row(i+1, cell_format=fmt_obj)
    
    if uw_account_flag == False:
        # set column width
        ws.set_column('A:A', 20)
        ws.set_column('B:N', 13)


