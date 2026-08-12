# Delayed and Scheduled Task Execution

Not all queued work should run immediately. A delay queue holds a task until a specified point in time and only makes it visible to workers once that time has passed; a worker that polls the queue before the delay elapses simply will not see the task, no matter how often it asks. This is how a system implements running a job at a specific future time without a dedicated scheduler thread sitting idle and counting down: the queue itself enforces the visibility window, and any worker that happens to poll after the deadline picks the task up.

Cron-like scheduling builds on the same primitive but recurs: instead of a single delayed task, the scheduler computes the next fire time from a cadence expression and re-enqueues a fresh delayed task each time the previous one runs, so a job that is supposed to run every hour never needs more than one pending delayed entry at once.

A naive implementation of recurring schedules has a failure mode of its own: if many jobs share the same cadence, such as a fleet of nightly reports all configured to fire at midnight, they all become eligible to run in the same instant, and every worker in the fleet tries to pick them up at once. This is a thundering herd, and it can spike load on downstream systems far beyond what steady-state traffic would produce.

The fix is jitter: instead of firing a scheduled task at exactly its configured time, the scheduler adds a small random offset, spreading what would have been one instant of load across a window of a few seconds to a few minutes. The size of that window is itself a tuning knob: too small and the herd barely spreads out, too large and time-sensitive jobs become sloppy about their own deadline.

Scheduled tasks that a caller cancels before their fire time still need to be removed from wherever the delay queue tracks pending entries, or a stale entry will fire anyway once its time arrives.
