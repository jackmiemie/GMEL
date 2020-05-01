# base code

TD_Modeling_0.2.4

# 1.0.0 features

- for AAAI2020 paper.

> focus on 2015 dataset
>
> separate embeddings.
>
> multitask in/out flow loss.
>
> Pre-training.

# 0.2 features

- **Combine with separate embeddings.**

- more resonable multitask in/out flow loss.

- smaller size of embedding for pre-training.

- pre-training

# 0.1 features

- GAT considers self loops.

> minor refinement: self-loop special treatment.


- rewrite the code structure for better extensibility.

> easy change last regression layer: linear plan and FNN plan
>
> adjustable multitask learning weights
>
> easy switch between devices: cpu and cuda


- use multitask regression loss function which includes in/out flow count losses. see model.py -> get_loss()

- use GAT instead of GCN

- use sqrt scale of trip volume

> In training phase, the scaled trip volume will be served as target. see main.py -> training epoch
>
> In evaluation phase, trip volume will be switch back to the original scale, and the metrics are based on the original scale. see utils.py -> evaluate()

- easy switch bewteen GPU and CPU