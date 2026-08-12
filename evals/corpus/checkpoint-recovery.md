# Checkpointing and Crash Recovery

A long-running job periodically writes a checkpoint — a durable record of how far it has progressed — so a crash does not force the whole job to restart from the beginning. Recovery means loading the most recent checkpoint and replaying only the work that happened after it, rather than reprocessing the entire job history from scratch every time a worker restarts. The design question in every checkpointing scheme is how much recent progress a worker is willing to redo in exchange for how rarely it pays the cost of writing a checkpoint at all.

## Snapshot interval

The snapshot interval sets an upper bound on how much log must be replayed after a restart, since a shorter interval means less replay work at recovery time but more steady-state overhead from writing checkpoints more often. Most systems tune the interval to keep replay under a few seconds rather than minimizing it outright.

## Durability

A checkpoint only counts as durable once a quorum of replicas has confirmed writing it, not merely after the local write returns, since a single-node acknowledgment would be lost if that node failed before the checkpoint replicated anywhere else. A recovery coordinator waits for several consecutive missed heartbeat checks before concluding a worker has actually crashed rather than merely paused, since a single skipped check is just as likely to be a garbage-collection pause or a slow network hop as it is an actual failure.

## Lease ownership after a crash

Checkpointed work is normally protected by the same lease a worker holds while processing it, and recovery branches on whether that lease was still active at the moment of the crash. If the crashed worker still held its lease, a successor must wait for the lease to lapse before resuming the checkpoint. If the lease had already expired before the crash, a waiting worker can resume the checkpoint immediately without any additional wait. Once a new owner takes over, it is issued a fencing token so the original worker cannot resume writing to the same checkpoint if it wakes up and restarts unexpectedly.

## Exactly-once side effects

Replaying a checkpointed log segment must not duplicate an exactly-once side effect, such as a payment already committed before the crash; replay logic has to tell "this step already happened" apart from "this step needs to run now."
