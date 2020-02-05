# automated-feedback-lab-sample

This repo contains the tracked data on one lab assignment in a programming course, where automated testing and feedback is deployed.

Two files are included:

* labCommitTableHashed: This file contains commits data
* labRepoTableHashed: This file contains repo data

Columns in labRepoTableHashed:

* timestamp_fp: The timestamp of when they clone the repository (i.e. their first push).
* timestampe_sp: The timestamp of their first commit to the repository (i.e. their second push).
* percent_passed: The number of test cases they were able to pass by the deadline.
* day_1: the signal of struggle per day - If the student fails to pass new test cases in their last 3 commits for the given day then this field, then value 1; otherwise 0
* day_2: same as above
* ...
* day_n: same as above
* repo_url_hash: hashed url of the repo

Columns in labCommitTableHashed:

* timestamp: timestamp of a commit
* comment: comment associated with a commit
* n_additions: number of newly added lines of code
* n_deletions: number of delete lines of code
* changes: number of changed lines of code
* n_passed: number of passed test cases
* n_run: number of test cases that were run (-1 representing compiling errors or exceptions)
* percent_passed: percentage of test cases that were passed
* repo_url_hash: hashed url of the repo where the commit was pushed to
