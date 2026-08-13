# Distributed Locks for Mutual Exclusion

A distributed lock keeps two workers in a cluster from processing the same job, row, or file at the same time. Unlike a lock inside a single process, a distributed lock has to survive network partitions, crashed holders, and clock drift between machines, so most production implementations back the lock with a time-bounded lease rather than a simple boolean flag.

## Lease-based acquisition

A worker asks the lock service for ownership of a named resource and, if granted, receives a lease with a fixed time-to-live. The client is only the legitimate holder for as long as that lease stays valid. A client keeps its exclusive hold only by renewing the lease before it expires, typically on a background timer set well inside the lease window so a single slow renewal request does not cause an accidental loss of ownership. If the holder crashes or is partitioned away from the lock service, the lease simply runs out and the resource becomes acquirable again without any manual cleanup step.

In a replicated deployment the lock service is not itself a single point of failure: acquisition also needs acknowledgment from a quorum of replicas before the lease is considered granted, so a client is never told it holds a lock that a partitioned minority of replicas never recorded.

## Fencing tokens

Lease timing alone is not quite safe. A client can pause — a long garbage collection cycle is the classic cause — for longer than its lease window, wake up still believing it holds the lock, and send a write after a second client has already acquired it. The fencing token lets storage reject any write tagged with an older token than the current holder's, closing this gap without requiring the storage layer to know anything about leases or timers at all. Every successful acquisition increments the token, and downstream systems simply refuse writes carrying a token lower than the highest one they have already accepted.

## Release

A well-behaved holder releases the lock explicitly once its critical section finishes, which is faster than waiting out the remaining lease, but the lease is what guarantees correctness even when a holder never gets the chance to release cleanly.
