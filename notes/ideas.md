## Current ideas to solve training balance
### Problem:
- there is tradeof between 
    - training too slow: live extraction of windows kills training process
    - training fast but not enough memory
### Idea:
- #### chunk loading:
    - prepare extracted windows( extracted out of multiple matrices as fixed datasets to be loaded)
- #### one matrix at a time:
    extract an entire big matrix using unfold (should fit exactly once)

