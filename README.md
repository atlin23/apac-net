# APAC-Net
APAC-Net is an algorithm for solving high-dimensional, and optionally stochastic, Mean-Field Games.

# Paper
The accompanying paper: https://arxiv.org/abs/2002.10113

If you found our paper or code helpful, please consider citing:

```tex
@article{lin2020apac,
  title={APAC-Net: Alternating the population and agent control via two neural networks to solve high-dimensional stochastic mean field games},
  author={Lin, Alex Tong and Fung, Samy Wu and Li, Wuchen and Nurbekyan, Levon and Osher, Stanley J},
  journal={arXiv preprint arXiv:2002.10113},
  year={2020}
}
```

# Usage
In order to start training, do

```bash
python main_apac-net.py
```

Inside `main_apac-net.py` there are hyperparameter that one can choose for solving the environment. The environment can be choseb by giving the proper name to `env_name`. Current options are `BottleneckCylinderEnv`, `TwoDiagCylinderEnv`, and `QuadcopteEnv`.

Once training is finished, one can run tests of the trained model by

```bash
python start_test.py
```

and giving the correct path for the `experiment_path` argument.
