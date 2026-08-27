We want a new downstream task that measures how well embeddings can be used to find optimal strategies for games.

The downstream task is as follows:
Given a set of (embedding, policy pairs) for player 1, and respectively for player 2, and an embedding->policy function for P1 and P2.

Train a "downstream task B" value function, which takes in a P1 embedding and a P2 embedding and outputs the expected value [extension: also takes in a PBS] if each player plays the corresponding policy. We train using supervised value targets (either ground truth or by sampling N times).

Then, do gradient descent-ascent in embedding space (two-timescale, probably, although alternatively we could try time-averaging in embedding space), by simply taking the gradient of the value function with respect to the P2 embedding (multiple times), and then taking the gradient of the value function with respect to the P1 embedding (opposite sign). [alternatively, use a zero-shot BR for P2]

Optionally, but probably importantly, for each P1, P2 embedding pair that we ever put into the value function, we should compute a target for that and use it to train the value function. We can call the first phase of training the value function "pretraining" and call this additional training "posttraining" (? name subject to change). 

We could try this with matching pennies and RPS. We could also try to design EFG games that resemble matching pennies and RPS, 


---

We need different LR for different embedding spaces. ORRRR we could just make all embedding spaces normalized.

produce plot of the embedding path but where we show the exploitability according to our value function (within the range of in-distribution embeddings).

