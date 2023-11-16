# PatrollingOptimization

This simulator serves as the training and evaluation platform in the following work:


> A Risk-aware Multi-objective Patrolling Route Optimization Method using Reinforcement Learning </br>
> Haowen Chen, Yifan Wu, Weikun Wang, Binbin Zhou </br>
> [IEEE ICPADS 2023 presentation](https://arxiv.org/abs/1802.06444)

### Prerequisites

- Python 2
- numpy
- pandas

### Run

```
cd ./tests/
python run_example.py
```

If you want to run your own map data, you need to change the input file before you run it. You have to normalize your road data, talk about replacing the crime and distance matrices in the tests folder as well as changing the map size parameter in code.

