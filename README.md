# VMG

We provide a complete Python implementation and experimental results of our Algorithm 2 on randomly generated MDPs for two-player zero-sum Markov games under the linear mixture model setting, where the python code and the curve of duality gap ($\max_{\pi_1}V^{\pi_1,\pi_{2,t}}(s_0)-\min_{\pi_2}V^{\pi_{1,t},\pi_2}(s_0)$ at the $t$-th iteration, c.f. (3) in our paper) v.s. iteration numbers are provided. In the experiments:
1. We adopt linear mixture model (c.f. Assumption 4): we randomly generate the feature function $\phi$ and $\theta^\star$ - the underlying parameter for the transition kernel of the MDP (i.e., $P_h(s'|s,a) = \phi_h(s,a,s')^\top \theta_h^\star$ for each $h\in[H]$). The reward function $r= (r_1,r_2): S\times A\rightarrow R_+^2$ is also randomly generated, with $r_2=-r_1$.

2. We solve line 4 (find NE of the game) in our Algorithm 2 using OMWU - Algorithm 2 in [Cen et al., 2022]. We compute $f_t$ in line 5 by minimizing the objective with 100 Adam steps.


Duality gap v.s. Number of iterations (Averaged over 3 random seeds) on randomly generated Markov games.
![Duality gap v.s. Number of iterations (Averaged over 3 random seeds)](vmg_avg_duality_gap.png)

Our empirical results demonstrate that the duality gap decreases significantly over iterations, confirming that the VMG a good Nash Equilibrium approximation within a reasonable number of iterations. 
