# -*- coding: utf-8 -*-
"""
Created on Sun Feb 05 14:26:59 2017

@author: alban.fauchere
"""

import numpy as np
import pandas as pd

    
# construct CP table, Named only, using Company Name as key (note that some trimming/normalization may have to be done)
# inverting sequence of portfolios (keep first Company, Portfolio overriding BKG)
def construct_cp_table(ptf_df, bkg_ptf_df):
    cp_table_df = ptf_df[ptf_df['Cardinality']==1].copy()
    # limit to relevant column
    cp_table_df = cp_table_df[['Company Name', 'Country', 'RiskFactor', 'Rating', 'PD', 'R']]
    # now bkg
    bkg_cp_table_df = bkg_ptf_df[bkg_ptf_df['Cardinality']==1].copy()
    bkg_cp_table_df = bkg_cp_table_df[['Company Name', 'Country', 'RiskFactor', 'Rating', 'PD', 'R']]
    # put together and keep first (use ptf value and only add bkg is different)
    comb_cp_table_df = cp_table_df.append(bkg_cp_table_df, ignore_index=True)
    #comb_cp_table_df[comb_cp_table_df.duplicated('Company Name')]
    comb_cp_table_df.drop_duplicates('Company Name', keep='first', inplace=True)
    return comb_cp_table_df


# make ptf_df consistent, Named and unnamed
def make_ptf_df_cp_consistent(ptf_df):
    # named and unnamed
    cp_table_df = ptf_df[['Company Name', 'Country', 'RiskFactor', 'Rating', 'PD', 'R', 'Cardinality', 'LNL']].copy()
    # drop duplicates, keeping first
    cp_table_df.drop_duplicates('Company Name', keep='first', inplace=True)
    # update ptf_df
    new_ptf_df = ptf_df.drop(['Country', 'RiskFactor', 'Rating', 'PD', 'R', 'Cardinality', 'LNL'], axis=1)
    new_ptf_df = new_ptf_df.merge(cp_table_df, on='Company Name', how='left')
    
    return new_ptf_df, cp_table_df

    
    