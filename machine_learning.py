from sklearn.linear_model import LinearRegression
from sklearn import tree
import graphviz
import operator
import time
from collections import namedtuple
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sortedcontainers import SortedDict

def load_data():
    data_format = '%Y-%m-%dT%H:%M:%SZ'
    CHART_STYLE = 'dark_background'
    START_DATE = datetime.strptime('2019-10-09T00:00:00Z',data_format)
    DAYS = 15
    
    #repo_url = 'https://github.com/wwu-csci-497/automated-feedback-lab-sample/blob/master/lab3_repo.csv'
    #commit_url = 'https://github.com/wwu-csci-497/automated-feedback-lab-sample/blob/master/lab3_commit.csv'
    repo_url = 'lab3_repo.csv'
    commit_url = 'lab3_commit.csv'
    
    repo_table = pd.read_csv(repo_url)
    print(f'loaded {len(repo_table)} repositories')
    commit_table = pd.read_csv(commit_url)
    print(f'loaded {len(commit_table)} commits')
    
    
    ########################## preprocess data ##############################
    def check_timestamp(s):
        try:
            datetime.strptime(s, data_format)
        except:
            print(f'Timestamp check failed on {s}')
            return False
        return True
    
    def get_day(s):
        date = datetime.strptime(s, data_format)
        days = (date - START_DATE).total_seconds()/(3600*24)
        if days > DAYS:
            raise ValueError()
        return days
    Commit = namedtuple(
            'Commit',
            ['repo_id','day','timestamp','comment','n_additions','n_deletions',
             'test_ratio']
    )
    Repo = namedtuple(
            'Repo',
            ['repo_id','init_day','start_day','test_ratio','struggled_on_day']
    )
    print(f'Preprocessing commits')
    commits = []
    for commit_row in commit_table.itertuples():
        if not check_timestamp(commit_row.timestamp):
            print(f'Commit {commit_row} has None timestamp')
            continue
        try:
            repo_id = commit_row.repo_url_hash
        except:
            repo_id = commit_row.repo_url
    
        try:
            commit = Commit(
                    repo_id=repo_id, day=get_day(commit_row.timestamp),
                    timestamp=commit_row.timestamp,comment=commit_row.comment,
                    n_additions=int(commit_row.n_additions),
                    n_deletions=int(commit_row.n_deletions),
                    test_ratio=(float(commit_row.n_passed)/float(commit_row.n_run)
                        if commit_row.n_run != 0 else 0)
            )
            if not commit.comment == 'Initial commit' and not commit.n_additions == 556:
                commits.append(commit)
        except ValueError:
            continue
    
    print(f'Preprocessing repos')
    repos = []
    for repo_row in repo_table.itertuples():
        timestamp_fp = repo_row.timestamp_fp
        timestamp_sp = repo_row.timestamp_sp
    
        if timestamp_fp == 'None':
            timestamp_fp = '2019-10-9T00:00:00Z'
        if timestamp_sp == 'None':
            timestamp_sp = '2019-10-9T00:00:00Z'
    
        try:
            repo_id = repo_row.repo_url_hash
        except:
            repo_id = repo_row.repo_url
        try:
            struggle_on_day = [
                    getattr(repo_row,f'day_{i+1}') == 1 for i in range(DAYS -1)
            ]
            repos.append(Repo(
                repo_id=repo_id,init_day=get_day(timestamp_fp),
                start_day=get_day(timestamp_sp),
                test_ratio=float(repo_row.percent_passed),
                struggled_on_day=struggle_on_day
            ))
        except ValueError:
            continue
    
    def epoch_to_day(t):
        return (datetime.fromtimestamp(t) - START_DATE).total_seconds() / (3600*24)
    
    
    ############################################ plot stuff ##################################################
    # order by repo name
    repo_name_hash = {}
    for commit in commits:
        if not commit.repo_id in repo_name_hash:
            repo_name_hash[commit.repo_id] = [commit]
        else:
            repo_name_hash[commit.repo_id].append(commit)
    
    repos = []
    x = []
    y = []

    # plot total number of changes vs total number of test casses passed
    for repo in repo_name_hash:
        repos.append(repo)
        commit_lst = repo_name_hash[repo]

        total_changes = 0
        total_additions = 0
        total_deletions = 0

        best_test_ratio = 0
        # build up total_changes and get target best_test_ratio
        for commit in commit_lst:
            total_additions += commit.n_additions
            total_deletions += commit.n_deletions
            total_changes = total_additions + total_deletions
            x.append(np.array([commit.day,total_changes,total_additions,total_deletions]))
            best_test_ratio = max(best_test_ratio, commit.test_ratio)
        y += ([best_test_ratio] * len(commit_lst))

    return np.array(x),np.array(y)


def train(x,y):
    reg=LinearRegression().fit(x,y)
    print(reg.score(x,y))
    clf = tree.DecisionTreeRegressor().fit(x,y)
    print(clf.score(x,y))
    tree.plot_tree(clf)
    feature_names=['day','n_changes','n_additions','n_deletions']

    data=tree.export_graphviz(clf,
                              out_file=None,
                              feature_names=feature_names,
                              class_names=['bullshit'],
                              filled=True,
                              rounded=True)
    graph = graphviz.Source(data,format='png')
    d = 10
    graph.render('./results/ds_{}'.format(d))
    

def main():
    x,y=load_data()
    train(x,y)

if __name__ == '__main__':
    main()
