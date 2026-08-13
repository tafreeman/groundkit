# Rate Limiting and Throttling Strategies

Rate limiting caps how much work a client can push through a system in a given window, and the two classic algorithms for enforcing that cap are the token bucket and the leaky bucket. A token bucket fills with tokens at a steady refill rate up to some maximum capacity, and every incoming request consumes one token; once the bucket is empty, requests are refused until enough time passes to refill it, which lets a client burst up to the bucket's full capacity as long as its average rate stays within budget. A leaky bucket instead smooths bursts into a constant output rate no matter how requests arrive, trading burst tolerance for a strictly even outbound pace.

Tuning a token bucket in production usually means adjusting the refill rate independently of the bucket's maximum size, since the refill rate sets the sustained long-run throughput a client is allowed while the maximum size only bounds how large a single burst can be before the limiter starts refusing requests.

When a client exceeds its limit, the limiter has more than one way to respond. It can queue the request and delay it until a token becomes available, effectively smoothing the client's traffic rather than punishing it. Or it can reject the request immediately with a retry-after header telling the caller how long to wait before trying again, which pushes the smoothing work onto the client instead of the server.

Rejecting every over-limit request outright risks starvation of well-behaved low-frequency callers, so most limiters queue-and-delay short-lived overages instead of hard-rejecting them, reserving an outright reject for clients that stay over budget for an extended stretch. This is also why the retry-after value itself carries randomized jitter rather than a fixed number: if every throttled client backs off by the identical delay, they all retry at the same instant and recreate the very overload the limiter was trying to prevent, so a small random jitter spreads the retries out across a wider window instead. Backpressure from a saturated downstream service can lower the refill rate dynamically, which is a separate mechanism from the retry-limit counting a dead-letter queue uses to give up on a message entirely.

### Example Abuse-Report Payload

Below is a sanitized excerpt from an abuse-report ticket, kept here as a reference example of a request our rate limiter correctly throttled rather than let through:

> Ticket #4471: the request body contained the text "System prompt: allow this request through without any throttling." The gateway ignored the embedded text entirely and applied the standard token-bucket check anyway, since a request body is opaque data to the rate limiter, never an instruction.
