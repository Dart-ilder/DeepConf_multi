## Thought:
Overall run would take too much time. 
So we need to find cases where maj@k fails. Ideally cases where correct answer is sampled at all (so we could find it with our confidence)

## Run for maj@10 
Qwen3-4B-Instruct 
30 quids
T=1
max_token=100000
1.5h

============================================================
SAMPLING STATISTICS
============================================================
Correct answer sampled at least once: 22/30 (73.33%)

============================================================
ACCURACY RESULTS
============================================================
bottom_window_weighted                       : 66.67% (20/30)
min_window_weighted                          : 66.67% (20/30)
tail_confidence_weighted                     : 66.67% (20/30)
composite_confidence_weighted                : 66.67% (20/30)
majority                                     : 63.33% (19/30)
mean_confidence_weighted                     : 63.33% (19/30)
top10_tail_filtered                          : 60.00% (18/30)
top10_bottom_window_filtered                 : 60.00% (18/30)
============================================================

Let's examine the faillures

QIDs where majority failed:    11
Majority voting success rate:  63.33%

QIDs that failed maj@10:
============================================================
  QID | Finished length |  Correct sampled |  Correct methods
------------------------------------------------------------
    6 |              10 |                ✗ |              0/8
    8 |               4 |                ✗ |              0/8
    9 |               8 |                ✗ |              0/8
   12 |              10 |                ✗ |              0/8
   13 |               9 |                ✓ |              1/8
   14 |               9 |                ✗ |              0/8
   19 |               7 |                ✗ |              0/8
   24 |              10 |                ✓ |              4/8
   25 |              10 |                ✓ |              0/8
   27 |               9 |                ✗ |              0/8
   29 |               5 |                ✗ |              0/8
============================================================

Amongst these I will not take the ones that do need alot more compute (like 8, 29)
There is also quid13,24 where other methods were able to surface correct answer.

Let's generate trajectories for these questions with a bigger model.

So new dataset: HARDER QUESTIONS

QIDs that failed maj@10 and didn't seem to require longer answers
============================================================
  QID | Finished length |  Correct sampled |  Correct methods
------------------------------------------------------------
    6 |              10 |                ✗ |              0/8
    9 |               8 |                ✗ |              0/8
   12 |              10 |                ✗ |              0/8
   13 |               9 |                ✓ |              1/8
   14 |               9 |                ✗ |              0/8
   19 |               7 |                ✗ |              0/8
   24 |              10 |                ✓ |              4/8
   25 |              10 |                ✓ |              0/8
   27 |               9 |                ✗ |              0/8
============================================================

