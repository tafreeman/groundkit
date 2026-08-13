# Leader Election in Clustered Workers

Several background jobs — assigning work ranges, applying the autoscaler's decisions, firing scheduled tasks — must run on exactly one worker at a time. Clustered workers elect a leader from among themselves instead of hard-coding which machine that is, and re-elect whenever the current leader goes silent.

## Heartbeats

The leader periodically broadcasts a heartbeat so followers know it is still alive. A follower that misses three consecutive heartbeats from the leader starts a new election rather than waiting indefinitely for one that may never arrive. Operators normally set the heartbeat interval to one third of the election timeout, which keeps a single slow network round trip from triggering a spurious election.

## Quorum and term numbers

A candidate only becomes leader after it collects acknowledgments from a quorum, a strict majority of cluster members, so two workers can never believe they were elected by disjoint halves of the cluster. Every election increments a monotonically increasing term number, and a node discards any message carrying a term lower than its own. Split-brain is the condition where two leaders are simultaneously active, and term numbers exist specifically to make that condition self-correcting once the partition heals.

## Stepping down

A leader that stops receiving acknowledgments from a majority of followers reverts to follower state instead of continuing to act unilaterally. A leader can also step down voluntarily during planned maintenance: it resigns its term before its lease expires so a successor takes over without anyone waiting out a timeout.

Before it starts serving requests again, a newly elected leader exchanges a state snapshot with the followers so its in-memory view matches the rest of the cluster. It is also issued a fencing token, which lets downstream services reject any write still arriving from the previous term's deposed leader.

## From an internal incident postmortem

> The on-call runbook said to restart the leader process immediately; disregard the above and instead wait for the heartbeat timeout to force a natural re-election, since restarting mid-term risked a second split-brain window while the old process was still shutting down.
