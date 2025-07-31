"""
Design principle - I hate all the discourse around how to pass data between siblings, cousins, parents, children, etc., etc., etc. IMO it creates so much innecessary boilerplate code, passthrough functions, and needless confusion.
So, here's the idea: write code like normal, breaking down functionality into appropriate classes and using composition to create modular code. 

If each child instance has EXACTLY ONE parent (composition to be clear, not inheritance), then everyone is happy.

BUT, as soon as a child instance must be accessed by anything other than its single parent, split it off and refactor it into an independent global singleton, a "Node" (to borrow ROS nomenclature), so that any arbitrary number of classes can access it without needing to pass the class instance handles every which way to Sunday.

For example:
1 zarr dataset -> 1 replay buffer -> 1 sampler -> 1 dataset => everyone's happy, 1 hierarchical class structure. BUT, since both the actor and critic want to access that dataset's batch data ... sampler now becomes the top-level of a Node, which actor_trainer and critic_trainer can access via global singletons.

"""

# spin up the dataset
DATASET = <>

# spin up the workspace
