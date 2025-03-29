import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
import random # For seeding
import math
import matplotlib.pyplot as plt
import os # For creating directories
import pickle # For saving/loading data

"""# Helper functions"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper Functions ---

def project_simplex(v, z=1.0):
    """Projects a vector v onto the probability simplex of sum z."""
    # Ensure input is contiguous before sorting if needed, although sort handles it
    # Check for non-finite values which can cause issues
    if not torch.isfinite(v).all():
        # print("Warning: Non-finite values detected in project_simplex input. Replacing with zeros.")
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) # Replace non-finite with 0
        # Optionally, could distribute z uniformly if v becomes all zero
        if torch.all(v == 0):
             # Check for division by zero
             if v.shape[0] == 0: return torch.zeros_like(v)
             if z > 0 : return torch.full_like(v, z / v.shape[0])
             else: return torch.zeros_like(v) # If z <=0, project to zero

    n_features = v.shape[0]
    # Ensure v is 1D for sorting
    v_flat = v.squeeze()
    if v_flat.dim() > 1: raise ValueError("project_simplex input must be 1D or squeezable to 1D")
    v = v_flat # Use the flattened version
    if n_features == 0: return torch.zeros_like(v) # Handle empty tensor


    # Sort v in descending order
    u, _ = torch.sort(v, descending=True) # _ ignores the indices
    cssv = torch.cumsum(u, dim=0)
    ind = torch.arange(1, n_features + 1, device=v.device, dtype=v.dtype) # ind should be 1, 2, ..., n
    cond = u * ind > (cssv - z) # Correct condition derived from paper/standard algorithms

    # FIX: Handle case where cond is empty or rho calculation fails
    cond_indices = torch.where(cond)[0]
    if len(cond_indices) == 0:
       # This case implies the projection might be onto a face or corner.
       # A common scenario: all elements are <= 0, and z > 0.
       # Robust fallback: put all mass on the max element if z > 0.
       # Another scenario: all elements > z/n. Project to uniform z/n.
       # Safest: if z>0 and n>0, project to max element or uniform? Let's stick to max.
       if z > 0 and n_features > 0:
           w = torch.zeros_like(v)
           max_idx = torch.argmax(v)
           w[max_idx] = z
           # print(f"Warning: project_simplex cond empty. Fallback to max element. Input: {v}, z:{z}")
           return w
       else: # If z<=0, project to zero vector
           # print(f"Warning: project_simplex cond empty and z<=0. Fallback to zero. Input: {v}, z:{z}")
           return torch.zeros_like(v)

    rho_idx = cond_indices[-1] # Get the index of the last True value
    rho = rho_idx.item() + 1 # Convert 0-based index to 1-based rho (number of positive elements)

    # Correct theta calculation using rho
    theta = (cssv[rho_idx] - z) / rho # cssv is 0-indexed

    w = torch.clamp(v - theta, min=0)
    # Renormalize to ensure sum is exactly z, handling potential floating point errors
    w_sum = w.sum()
    # Use a tolerance appropriate for the data type
    tol = 1e-6 if v.dtype == torch.float32 else 1e-8
    if abs(w_sum - z) > tol: # If sum is not close enough to z
        if w_sum > tol : # Avoid division by zero or amplifying tiny errors
            w = w * (z / w_sum)
        elif z > 0 and n_features > 0: # If sum is near zero, but z>0, likely projection is to a corner -> put mass on max
             # print(f"Warning: project_simplex renormalization needed, w_sum near zero. Fallback to max. Sum: {w_sum}")
             w = torch.zeros_like(v)
             max_idx = torch.argmax(v)
             w[max_idx] = z
        else: # If z<=0 and sum is near zero, return zero vector
             # print(f"Warning: project_simplex renormalization needed, w_sum near zero, z<=0. Fallback to zero. Sum: {w_sum}")
             w = torch.zeros_like(v)

    # Final check for NaNs
    if torch.isnan(w).any():
        # print(f"Warning: NaN detected in project_simplex output. Input v: {v}, z: {z}")
        # Fallback to uniform distribution if possible
        if z > 0 and n_features > 0:
            return torch.full_like(v, z / n_features)
        else:
            return torch.zeros_like(v)

    return w


def generate_linear_mixture_mdp(H, S, A1, A2, d, seed=None):
    """
    Generates the components for a two-player zero-sum Markov game
    with linear mixture transitions and random rewards.
    Satisfies Assumption 4 and ensures valid transitions.
    phi shape: (H, S, A1, A2, S_next, d) -> h,s,a,b,n,i
    theta shape: (H, d) -> h,i
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    print(f"\n--- Generating MDP Components (Seed: {seed}) ---") # Indicate seed used

    phi = torch.zeros(H, S, A1, A2, S, d, device=device)
    base_kernels = torch.zeros(d, H, S, A1, A2, S, device=device) # Shape: i,h,s,a,b,n

    # 1. Generate d base valid transition kernels P_base^i
    # print("Generating base kernels...")
    for i in range(d):
        probs_unnorm = torch.rand(H, S, A1, A2, S, device=device)
        probs_unnorm = probs_unnorm + 0.1 * torch.rand_like(probs_unnorm)
        probs = F.softmax(probs_unnorm, dim=-1)
        base_kernels[i] = probs

    # 2. Construct phi features
    # print("Constructing phi features...")
    phi = base_kernels.permute(1, 2, 3, 4, 5, 0).contiguous()

    # Check feature constraints
    # print("Checking feature constraints...")
    phi_sum_check = torch.sum(phi, dim=4)
    max_dev = torch.max(torch.abs(phi_sum_check - 1.0)).item()
    if not torch.allclose(phi_sum_check, torch.ones_like(phi_sum_check), atol=1e-5):
         print(f"WARNING: Feature constraint sum phi != 1 failed! Max dev: {max_dev:.6e}")
    # else:
    #      print(f"Feature constraint sum phi = 1 check passed (Max dev: {max_dev:.6e}).")


    # 3. Generate true parameter theta_star from the simplex Delta^{d-1} for each h
    theta_star = torch.zeros(H, d, device=device)
    for h in range(H):
        theta_h_unproj = 10*torch.rand(d, device=device)
        theta_star[h] = project_simplex(theta_h_unproj, z=1.0)
        if not torch.allclose(theta_star[h].sum(), torch.tensor(1.0, device=device), atol=1e-6):
            # print(f"Warning: theta_star[{h}] does not sum to 1. Sum: {theta_star[h].sum()}. Re-projecting.")
            theta_star[h] = project_simplex(theta_star[h], z=1.0) # Try again


    # 4. Generate random rewards in [0, 1] for player 1
    r1 = 3*torch.rand(H, S, A1, A2, device=device)
    r = r1

    # 5. Verify the true transition kernel sums to 1 (Req 1)
    # print("\n--- Verifying True Transition Kernel (Req 1) ---")
    try:
        P_star = torch.stack([compute_transition(phi[h], theta_star[h]) for h in range(H)])
        P_star_sums = torch.sum(P_star, dim=-1)
        max_dev_p = torch.max(torch.abs(P_star_sums - 1.0)).item()
        if torch.allclose(P_star_sums, torch.ones_like(P_star_sums), atol=1e-5):
            pass # print(f"True transition kernel sum check passed (Max dev: {max_dev_p:.6e}).")
        else:
            print(f"ERROR: True transition kernel does not sum to 1! Max deviation={max_dev_p:.6e}")
    except Exception as e:
        print(f"Error during P_star computation/verification: {e}")

    # print("--- Verification End ---")
    print("--- MDP Generation Complete ---")
    return phi, theta_star, r


def compute_transition(phi_h, theta_h):
    """
    Computes P(n|s,a,b) for a given step h using phi_h and theta_h.
    phi_h shape: (S, A1, A2, S_next, d) -> s,a,b,n,i
    theta_h shape: (d,) -> i
    Output P_h shape: (S, A1, A2, S_next) -> s,a,b,n
    """
    theta_h_valid = theta_h.clamp(min=0) # Ensure non-negative
    P_h = torch.einsum('sabni,i->sabn', phi_h, theta_h_valid)
    P_h = torch.clamp(P_h, min=0) # Use exactly 0
    p_sum = torch.sum(P_h, dim=-1, keepdim=True)
    S_next_dim = P_h.shape[-1]
    uniform_prob = torch.ones_like(P_h) / S_next_dim if S_next_dim > 0 else torch.zeros_like(P_h)
    P_h = torch.where(p_sum > 1e-8, P_h / p_sum, uniform_prob)
    if torch.isnan(P_h).any():
        print(f"CRITICAL WARNING: NaN detected in compute_transition output!")
        P_h = torch.where(torch.isnan(P_h), uniform_prob, P_h)
    return P_h


@torch.no_grad()
def compute_q_v_fomwu(H, S, A1, A2, P_tensor, r_tensor, pi1, pi2, tau):
    """Computes Q and V functions using DP for FOMWU (QRE)."""
    V = torch.zeros(H + 1, S, device=device)
    Q = torch.zeros(H, S, A1, A2, device=device)
    safe_tau = max(tau, 1e-9)
    log_clamp_min = 1e-12

    # Ensure policies are valid probabilities (robust check)
    pi1 = pi1.clamp(min=0)
    pi1_sums = pi1.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    pi1 = pi1 / pi1_sums
    pi2 = pi2.clamp(min=0)
    pi2_sums = pi2.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    pi2 = pi2 / pi2_sums

    pi1_log = torch.log(pi1.clamp(min=log_clamp_min))
    pi2_log = torch.log(pi2.clamp(min=log_clamp_min))

    for h in range(H - 1, -1, -1):
        P_h = P_tensor[h] # s,a,b,n
        r_h = r_tensor[h] # s,a,b
        expected_next_V = torch.einsum('sabn,n->sab', P_h, V[h+1])
        Q[h] = r_h + expected_next_V
        pi1_h_s = pi1[h]
        pi2_h_s = pi2[h]
        Q_h_s = Q[h]
        expected_Q = torch.einsum('sa,sb,sab->s', pi1_h_s, pi2_h_s, Q_h_s)
        entropy_pi1 = torch.sum(pi1_h_s * pi1_log[h], dim=1)
        entropy_pi2 = torch.sum(pi2_h_s * pi2_log[h], dim=1)
        V[h] = expected_Q - safe_tau * entropy_pi1 + safe_tau * entropy_pi2
        if torch.isnan(V[h]).any():
            print(f"NaN detected in V[{h}] during compute_q_v_fomwu")
            V[h] = torch.nan_to_num(V[h], nan=0.0)

    return V, Q


def solve_ne_fomwu(H, S, A1, A2, phi, theta_fomwu, r, K_inner, fomwu_eta, fomwu_tau, fomwu_alpha_factor, start_state_dist=None):
    """Solves for the QRE using FOMWU."""
    # Initialize policies as uniform
    pi1_k = torch.ones(H, S, A1, device=device) / A1 if A1 > 0 else torch.zeros(H, S, A1, device=device)
    pi2_k = torch.ones(H, S, A2, device=device) / A2 if A2 > 0 else torch.zeros(H, S, A2, device=device)

    with torch.no_grad():
        theta_fomwu_valid = theta_fomwu.clone()
        for h in range(H):
            theta_fomwu_valid[h] = project_simplex(theta_fomwu[h], z=1.0)
        P_fomwu = torch.stack([compute_transition(phi[h], theta_fomwu_valid[h]) for h in range(H)])

    eta_tau_safe = min(max(fomwu_eta * fomwu_tau, 1e-9), 1.0 - 1e-9)
    safe_tau_div = max(fomwu_tau, 1e-9)
    log_clamp_min = 1e-12
    softmax_temp = 1.0
    prev_pi1_k = pi1_k.clone()
    prev_pi2_k = pi2_k.clone()

    for k in range(K_inner):
        with torch.no_grad():
             _, Q_for_update = compute_q_v_fomwu(H, S, A1, A2, P_fomwu, r, pi1_k, pi2_k, fomwu_tau)
             if torch.isnan(Q_for_update).any():
                 print(f"ERROR: NaN in Q_for_update at FOMWU step {k}. Returning previous policies.")
                 return prev_pi1_k, prev_pi2_k, float('nan') # Return NaN gap

        with torch.no_grad():
            # --- Policy Updates (Eq 11a, 11b) ---
            Q_pi2 = torch.einsum('hsab,hsb->hsa', Q_for_update, pi2_k)
            Q_pi1 = torch.einsum('hsab,hsa->hsb', Q_for_update, pi1_k)
            log_pi1_k = torch.log(pi1_k.clamp(min=log_clamp_min))
            log_pi2_k = torch.log(pi2_k.clamp(min=log_clamp_min))

            if torch.isnan(log_pi1_k).any() or torch.isnan(Q_pi2).any() or \
               torch.isnan(log_pi2_k).any() or torch.isnan(Q_pi1).any():
                print(f"ERROR: NaN before pi_bar update at FOMWU step {k}.")
                return prev_pi1_k, prev_pi2_k, float('nan')

            log_pi1_bar_unnorm = (1 - eta_tau_safe) * log_pi1_k + fomwu_eta * Q_pi2
            log_pi2_bar_unnorm = (1 - eta_tau_safe) * log_pi2_k - fomwu_eta * Q_pi1
            pi1_bar_k = F.softmax(log_pi1_bar_unnorm / softmax_temp, dim=-1)
            pi2_bar_k = F.softmax(log_pi2_bar_unnorm / softmax_temp, dim=-1)

            if torch.isnan(pi1_bar_k).any() or torch.isnan(pi2_bar_k).any():
                print(f"ERROR: NaN after pi_bar update at FOMWU step {k}.")
                return prev_pi1_k, prev_pi2_k, float('nan')

            Q_pi2_bar = torch.einsum('hsab,hsb->hsa', Q_for_update, pi2_bar_k)
            Q_pi1_bar = torch.einsum('hsab,hsa->hsb', Q_for_update, pi1_bar_k)
            log_pi1_bar_k = torch.log(pi1_bar_k.clamp(min=log_clamp_min))
            log_pi2_bar_k = torch.log(pi2_bar_k.clamp(min=log_clamp_min))

            if torch.isnan(log_pi1_bar_k).any() or torch.isnan(Q_pi2_bar).any() or \
               torch.isnan(log_pi2_bar_k).any() or torch.isnan(Q_pi1_bar).any():
                print(f"ERROR: NaN before pi_next update at FOMWU step {k}.")
                return prev_pi1_k, prev_pi2_k, float('nan')

            log_pi1_next_unnorm = (1 - eta_tau_safe) * log_pi1_bar_k + fomwu_eta * Q_pi2_bar
            log_pi2_next_unnorm = (1 - eta_tau_safe) * log_pi2_bar_k - fomwu_eta * Q_pi1_bar
            pi1_next = F.softmax(log_pi1_next_unnorm / softmax_temp, dim=-1)
            pi2_next = F.softmax(log_pi2_next_unnorm / softmax_temp, dim=-1)

            if torch.isnan(pi1_next).any() or torch.isnan(pi2_next).any():
                print(f"ERROR: NaN after pi_next update at FOMWU step {k}.")
                return prev_pi1_k, prev_pi2_k, float('nan')
            # --- End Policy Updates ---

        prev_pi1_k = pi1_k.clone()
        prev_pi2_k = pi2_k.clone()
        pi1_k = pi1_next
        pi2_k = pi2_next

    # --- Calculate Duality Gap for M(theta_fomwu) ---
    duality_gap = float('nan')
    with torch.no_grad():
        pi1_final = pi1_k
        pi2_final = pi2_k
        P_fomwu = P_fomwu.detach()
        br_val_p1, _ = compute_best_response_value_and_policy(
            H, S, A1, A2, 0, pi2_final, P_fomwu, r, fomwu_tau, start_state_dist, requires_grad=False
        )
        br_val_p2, _ = compute_best_response_value_and_policy(
            H, S, A1, A2, 1, pi1_final, P_fomwu, r, fomwu_tau, start_state_dist, requires_grad=False
        )
        if torch.isnan(br_val_p1) or torch.isnan(br_val_p2):
            print("  FOMWU Solver: NaN detected during duality gap calculation for M(theta_fomwu).")
        else:
            duality_gap = br_val_p1.item() - br_val_p2.item()

    return pi1_k, pi2_k, duality_gap

# Decorator context manager for conditional gradients
class CondGrad:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.prev_grad_enabled = torch.is_grad_enabled()
    def __enter__(self):
        if self.enabled: torch.set_grad_enabled(True)
        else: torch.set_grad_enabled(False)
    def __exit__(self, type, value, traceback):
        torch.set_grad_enabled(self.prev_grad_enabled)

def compute_best_response_value_and_policy(H, S, A1, A2, player_idx, fixed_pi, P_tensor, r_tensor, tau, start_state_dist=None, requires_grad=True):
    """Computes BR value AND policy, differentiable w.r.t P_tensor if requires_grad=True."""
    with CondGrad(requires_grad):
        V_br_steps = [torch.zeros(S, device=device) for _ in range(H + 1)]
        num_actions = A1 if player_idx == 0 else A2
        pi_br_steps = [torch.zeros(S, num_actions, device=device) for _ in range(H)]

        # Robust check for fixed_pi validity
        fixed_pi = fixed_pi.detach().clamp(min=0)
        fixed_pi_sums = fixed_pi.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        fixed_pi = fixed_pi / fixed_pi_sums

        log_clamp_min = 1e-12
        fixed_pi_log = torch.log(fixed_pi.clamp(min=log_clamp_min))
        safe_tau = max(tau, 1e-9)

        for h in range(H - 1, -1, -1):
            P_h = P_tensor[h]
            r_h = r_tensor[h].detach()
            expected_next_V = torch.einsum('sabn,n->sab', P_h, V_br_steps[h+1])
            current_Q = r_h + expected_next_V

            if player_idx == 0: # Player 1 (max)
                fixed_pi_h_s = fixed_pi[h]
                fixed_pi_log_h_s = fixed_pi_log[h]
                Q_avg_over_b = torch.einsum('sab,sb->sa', current_Q, fixed_pi_h_s)
                log_fixed_pi_term = torch.einsum('sb,sb->s', fixed_pi_h_s, fixed_pi_log_h_s)
                q_avg_max = torch.max(Q_avg_over_b, dim=-1, keepdim=True)[0]
                # Handle cases where Q_avg_over_b might be constant -> uniform softmax
                exp_term = torch.exp((Q_avg_over_b - q_avg_max) / safe_tau)
                pi_br_h = exp_term / exp_term.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                # pi_br_h = F.softmax((Q_avg_over_b - q_avg_max) / safe_tau, dim=-1) # Original

            else: # Player 2 (min)
                fixed_pi_h_s = fixed_pi[h]
                fixed_pi_log_h_s = fixed_pi_log[h]
                Q_avg_over_a = torch.einsum('sab,sa->sb', current_Q, fixed_pi_h_s)
                log_fixed_pi_term = torch.einsum('sa,sa->s', fixed_pi_h_s, fixed_pi_log_h_s)
                q_avg_max = torch.max(-Q_avg_over_a, dim=-1, keepdim=True)[0]
                # Handle constant Q_avg_over_a -> uniform softmax
                exp_term = torch.exp((-Q_avg_over_a - q_avg_max) / safe_tau)
                pi_br_h = exp_term / exp_term.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                # pi_br_h = F.softmax((-Q_avg_over_a - q_avg_max) / safe_tau, dim=-1) # Original

            pi_br_steps[h] = pi_br_h
            expected_Q_br = torch.sum(pi_br_h * (Q_avg_over_b if player_idx == 0 else Q_avg_over_a), dim=-1)
            entropy_br = torch.sum(pi_br_h * torch.log(pi_br_h.clamp(min=log_clamp_min)), dim=-1)

            if player_idx == 0:
                V_br_h = expected_Q_br - safe_tau * entropy_br + safe_tau * log_fixed_pi_term
            else:
                V_br_h = expected_Q_br - safe_tau * log_fixed_pi_term + safe_tau * entropy_br

            # NaN Checks and Fallbacks
            if torch.isnan(V_br_h).any():
                print(f"NaN detected in V_br_h at h={h} for player {player_idx}. Falling back.")
                V_br_h = torch.nan_to_num(V_br_h, nan=0.0)
            if torch.isnan(pi_br_steps[h]).any():
                 print(f"NaN detected in pi_br_steps[{h}] for player {player_idx}. Falling back.")
                 fallback_policy = torch.ones_like(pi_br_steps[h])
                 num_actions_fallback = fallback_policy.shape[-1]
                 if num_actions_fallback > 0: fallback_policy /= num_actions_fallback
                 pi_br_steps[h] = fallback_policy

            V_br_steps[h] = V_br_h

        if start_state_dist is None:
            start_state_dist = torch.ones(S, device=device) / S if S > 0 else torch.zeros(S, device=device)
        start_state_dist = start_state_dist.detach()

        expected_V_start = torch.sum(start_state_dist * V_br_steps[0]) if S > 0 else torch.tensor(0.0, device=device)
        best_response_pi = torch.stack(pi_br_steps) if H > 0 else torch.empty((0, S, num_actions), device=device)

        if torch.isnan(expected_V_start):
             print(f"NaN detected in final expected_V_start for player {player_idx}. Falling back.")
             expected_V_start = torch.tensor(0.0, device=device)
        if torch.isnan(best_response_pi).any():
             print(f"NaN detected in final best_response_pi for player {player_idx}. Falling back.")
             fallback_pi = torch.ones_like(best_response_pi)
             num_actions_fallback = fallback_pi.shape[-1]
             if num_actions_fallback > 0: fallback_pi /= num_actions_fallback
             best_response_pi = fallback_pi

        if not requires_grad:
            expected_V_start = expected_V_start.detach()
            best_response_pi = best_response_pi.detach()

    return expected_V_start, best_response_pi


@torch.no_grad()
def simulate_trajectory(H, S, A1, A2, pi1, pi2, phi, theta_sim, start_state=0):
    """Simulates one trajectory using policies pi1, pi2 and model (phi, theta_sim)."""
    s = start_state
    trajectory_data = []
    try:
        theta_sim_valid = theta_sim.clone()
        for h_p in range(H): theta_sim_valid[h_p] = project_simplex(theta_sim[h_p], z=1.0)
        P_sim = torch.stack([compute_transition(phi[h], theta_sim_valid[h]) for h in range(H)])
    except Exception as e:
         print(f"CRITICAL ERROR computing P_sim in simulate_trajectory: {e}")
         return []

    pi1 = pi1.detach()
    pi2 = pi2.detach()

    for h in range(H):
        # --- Validate and Sample Actions (Robust) ---
        pi1_h_s = pi1[h, s].clamp(min=0) # Non-negative
        pi1_sum = pi1_h_s.sum()
        if pi1_sum > 1e-8: pi1_h_s = pi1_h_s / pi1_sum
        elif A1 > 0: pi1_h_s = torch.ones_like(pi1_h_s) / A1
        else: pi1_h_s = torch.zeros_like(pi1_h_s)

        pi2_h_s = pi2[h, s].clamp(min=0) # Non-negative
        pi2_sum = pi2_h_s.sum()
        if pi2_sum > 1e-8: pi2_h_s = pi2_h_s / pi2_sum
        elif A2 > 0: pi2_h_s = torch.ones_like(pi2_h_s) / A2
        else: pi2_h_s = torch.zeros_like(pi2_h_s)

        try:
            # Use Categorical(logits=...) for potentially better stability if probs are tiny? Stick to probs for now.
            dist1 = Categorical(probs=pi1_h_s)
            a1 = dist1.sample()
            dist2 = Categorical(probs=pi2_h_s)
            a2 = dist2.sample()
        except ValueError as e:
            print(f"ERROR sampling actions at h={h}, s={s}. P1_probs: {pi1_h_s.cpu().numpy()} (sum={pi1_sum}), P2_probs: {pi2_h_s.cpu().numpy()} (sum={pi2_sum}). Error: {e}")
            # Fallback: Sample uniformly if sampling fails
            a1 = torch.randint(0, A1, (1,), device=device).item() if A1 > 0 else 0
            a2 = torch.randint(0, A2, (1,), device=device).item() if A2 > 0 else 0
            print(f"Falling back to uniform action sampling: a1={a1}, a2={a2}")
        # --- End Action Sampling ---

        # --- Validate and Sample Next State (Robust) ---
        try:
            next_s_dist_probs = P_sim[h, s, a1, a2].clamp(min=0) # Non-negative
            dist_sum = next_s_dist_probs.sum()
            if dist_sum > 1e-8: next_s_dist_probs = next_s_dist_probs / dist_sum
            elif S > 0: next_s_dist_probs = torch.ones_like(next_s_dist_probs) / S
            else: next_s_dist_probs = torch.zeros_like(next_s_dist_probs)
        except IndexError:
            print(f"CRITICAL INDEX Error accessing P_sim[{h},{s},{a1},{a2}]. Shape={P_sim.shape}")
            s_next = torch.randint(0, S, (1,), device=device).item() if S > 0 else 0
            trajectory_data.append((h, s, a1.item() if isinstance(a1, torch.Tensor) else a1,
                                    a2.item() if isinstance(a2, torch.Tensor) else a2, s_next))
            s = s_next
            continue

        try:
            dist_next_s = Categorical(probs=next_s_dist_probs)
            s_next = dist_next_s.sample()
        except ValueError as e:
            print(f"CRITICAL Error sampling next state at h={h}, s={s}, a1={a1}, a2={a2}.")
            print(f"Probs: {next_s_dist_probs.cpu().numpy()}, Sum: {dist_sum.item()}. Error: {e}")
            s_next = torch.randint(0, S, (1,), device=device).item() if S > 0 else 0
            print(f"Falling back to uniform next state sampling: s_next={s_next}")
        # --- End Next State Sampling ---

        trajectory_data.append((h, s, a1.item() if isinstance(a1, torch.Tensor) else a1,
                                a2.item() if isinstance(a2, torch.Tensor) else a2,
                                s_next.item() if isinstance(s_next, torch.Tensor) else s_next))
        s = s_next.item() if isinstance(s_next, torch.Tensor) else s_next

    return trajectory_data


def update_model_vmg(H, S, A1, A2, d, phi, current_theta, dataset, pi1_t, pi2_t, r, vmg_alpha, vmg_lr, model_optim_steps, fomwu_tau, start_state_dist):
    """Updates the model parameter theta using VMG update rule (Eq. 22)."""
    theta_optim = current_theta.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([theta_optim], lr=vmg_lr)

    pi1_t_detached = pi1_t.detach()
    pi2_t_detached = pi2_t.detach()

    # --- Precompute indices and counts for MLE ---
    mle_data = {}
    transition_counts = {}
    total_mle_samples = 0
    for h in range(H):
        for s, a1, a2, s_next in dataset[h]:
             if 0 <= h < H and 0 <= s < S and 0 <= a1 < A1 and 0 <= a2 < A2 and 0 <= s_next < S:
                 key = (h, s, a1, a2)
                 if key not in mle_data:
                     mle_data[key] = {}
                     transition_counts[key] = 0
                 mle_data[key][s_next] = mle_data[key].get(s_next, 0) + 1
                 transition_counts[key] += 1
                 total_mle_samples += 1

    if total_mle_samples == 0:
        print("  Warning: No data in dataset for model update. Skipping VMG update.")
        return current_theta

    log_interval = max(1, model_optim_steps // 5)
    last_valid_theta = current_theta.clone().detach()
    # print(f"  Starting model optimization ({model_optim_steps} steps, {total_mle_samples} samples)...")

    for step in range(model_optim_steps):
        optimizer.zero_grad()

        # Project theta_optim before loss computation for stability
        with torch.no_grad():
             theta_proj = theta_optim.clone()
             for h_proj in range(H):
                 theta_proj[h_proj] = project_simplex(theta_optim[h_proj], z=1.0)
        theta_proj.requires_grad_(True)
        current_theta_for_loss = theta_proj

        # --- Compute NLL ---
        P_tensors_optim_list = []
        nll = torch.tensor(0.0, device=device)
        nan_in_p = False
        for h_p in range(H):
            P_h_optim = compute_transition(phi[h_p], current_theta_for_loss[h_p])
            if torch.isnan(P_h_optim).any():
                 print(f"\n >>> Error: NaN in P_h_optim at VMG step {step}, h={h_p}. <<<")
                 nan_in_p = True
                 break # Stop computing P if NaN found
            P_tensors_optim_list.append(P_h_optim)

        if nan_in_p: # Skip NLL calculation if P had NaN
            print("    Skipping NLL due to NaN in P_h_optim.")
            return last_valid_theta

        for (h_key, s_key, a1_key, a2_key), sn_counts in mle_data.items():
            P_h_key = P_tensors_optim_list[h_key]
            for sn_key, count in sn_counts.items():
                try:
                    prob_sn = P_h_key[s_key, a1_key, a2_key, sn_key]
                    log_prob_sn = torch.log(prob_sn.clamp(min=1e-12))
                    nll = nll - count * log_prob_sn
                except IndexError:
                     print(f"Index Error during NLL calc: P_h_{h_key}[{s_key},{a1_key},{a2_key},{sn_key}]")
                     # Handle gracefully, maybe skip this sample? Or assign high NLL?
                     nll = nll + count * 100 # Penalize heavily?
                     print("    Assigning high NLL penalty.")


        if torch.isnan(nll) or not torch.isfinite(nll):
            print(f"\n >>> Error: NLL is NaN/inf at VMG optim step {step}. NLL={nll} <<<")
            return last_valid_theta
        nll = nll / total_mle_samples
        # --- NLL End ---

        # --- Compute Value Incentive ---
        P_tensors_optim_stacked = torch.stack(P_tensors_optim_list)
        if torch.isnan(P_tensors_optim_stacked).any():
             print(f"\n >>> Error: NaN in P_tensors_optim_stacked at VMG step {step}. <<<")
             return last_valid_theta

        br_val_p1, _ = compute_best_response_value_and_policy(H, S, A1, A2, 0, pi2_t_detached, P_tensors_optim_stacked, r, fomwu_tau, start_state_dist, requires_grad=True)
        br_val_p2, _ = compute_best_response_value_and_policy(H, S, A1, A2, 1, pi1_t_detached, P_tensors_optim_stacked, r, fomwu_tau, start_state_dist, requires_grad=True)
        value_incentive = br_val_p1 + br_val_p2

        if torch.isnan(br_val_p1) or torch.isnan(br_val_p2):
            print(f"\n >>> Error: NaN in BR values at VMG step {step}. BR1={br_val_p1}, BR2={br_val_p2} <<<")
            return last_valid_theta
        # --- Value Incentive End ---

        # --- Total Loss & Backprop ---
        loss = nll - vmg_alpha * value_incentive
        if not torch.isfinite(loss):
            print(f"\n >>> Error: Non-finite loss at VMG step {step}. Loss={loss.item()} <<<")
            print(f"    NLL={nll.item():.6f}, BR1={br_val_p1.item():.6f}, BR2={br_val_p2.item():.6f}, alpha={vmg_alpha}")
            return last_valid_theta

        loss.backward()

        if current_theta_for_loss.grad is None or not torch.isfinite(current_theta_for_loss.grad).all():
            print(f"\n >>> Warning: Non-finite grads on current_theta_for_loss at VMG step {step}. Loss={loss.item()} <<<")
            optimizer.zero_grad()
            theta_optim.requires_grad_(True) # Ensure grad tracking remains for next step
            continue # Skip optimizer step

        # Assign gradient from projected theta back to the optimizable theta
        if theta_optim.grad is None: theta_optim.grad = torch.zeros_like(theta_optim)
        theta_optim.grad.data = current_theta_for_loss.grad.data.clone() # Copy grad data

        # Clip gradients and optimize
        grad_norm = torch.nn.utils.clip_grad_norm_([theta_optim], max_norm=10.0)
        if not torch.isfinite(grad_norm):
             print(f"\n >>> Warning: Non-finite grad norm AFTER clipping ({grad_norm}) at VMG step {step}. Skipping step.")
             optimizer.zero_grad()
             theta_optim.requires_grad_(True)
             continue

        optimizer.step()

        # Project AFTER Adam step
        with torch.no_grad():
            for h_proj in range(H):
                theta_optim[h_proj] = project_simplex(theta_optim[h_proj], z=1.0)

        # Ensure grad tracking for next iteration
        theta_optim.requires_grad_(True)

        # Logging (optional, can be verbose)
        # if step % log_interval == 0 or step == model_optim_steps - 1:
        #      print(f"    step {step:4d}: loss={loss.item():+.6f} (NLL: {nll.item():.6f}, ValInc: {value_incentive.item():+.6f}), GradNorm: {grad_norm:.4f}")

        last_valid_theta = theta_optim.clone().detach()

    # Return the updated theta after all optimization steps
    final_theta = theta_optim.detach()
    # Final check and projection
    with torch.no_grad():
        for h in range(H):
            if not torch.allclose(final_theta[h].sum(), torch.tensor(1.0, device=device), atol=1e-5):
                print(f"Warning: Final theta_optim[{h}] sum is {final_theta[h].sum()}. Re-projecting.")
                final_theta[h] = project_simplex(final_theta[h], z=1.0)
            if torch.isnan(final_theta[h]).any():
                 print(f"CRITICAL Error: NaN in final theta_optim[{h}]. Resetting to uniform.")
                 final_theta[h] = torch.ones(d, device=device) / d if d > 0 else torch.zeros(d, device=device)

    return final_theta

"""# Main VMG Algorithm"""

# --- Main VMG Algorithm ---
def run_vmg(H, S, A1, A2, d, T, K_fomwu_inner, vmg_alpha_base, vmg_lr, model_optim_steps, fomwu_eta, fomwu_tau, fomwu_alpha_factor, seed=None):
    """Runs the VMG algorithm."""

    print(f"\n===== Running VMG with Seed: {seed} =====")
    # Set seed for this run
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    print("Initializing...")
    phi, theta_star, r = generate_linear_mixture_mdp(H, S, A1, A2, d, seed=seed)

    # Initial model estimate (uniform)
    theta_t = torch.ones(H, d, device=device) / d if d > 0 else torch.zeros(H, d, device=device)
    for h_init in range(H): theta_t[h_init] = project_simplex(theta_t[h_init], z=1.0)
    theta_t = theta_t.requires_grad_(False)

    dataset = [[] for _ in range(H)]
    start_state_dist = torch.ones(S, device=device) / S if S > 0 else torch.zeros(S, device=device)

    # Precompute P_star for true duality gap calculation
    with torch.no_grad():
        P_star_tensor = torch.stack([compute_transition(phi[h], theta_star[h]) for h in range(H)])

    thetas_history = [theta_t.cpu().numpy()]
    initial_error = torch.norm(theta_t - theta_star).item()
    theta_errors = [initial_error]
    duality_gaps_history = [] # Store duality gap of pi_t in M(theta*) from t=1 to T

    print(f"Initial Theta Error ||theta_0 - theta*||: {initial_error:.4f}")
    print("Starting VMG iterations...")
    start_time = time.time()

    for t in range(1, T + 1):
        iter_start_time = time.time()
        print(f"\n--- VMG Iteration {t}/{T} (Seed: {seed}) ---")
        current_vmg_alpha = vmg_alpha_base # Fixed alpha

        # --- 1. Determine Policy pi_t ---
        # **** MODIFICATION: Remove special case for t=1 ****
        # Always solve for the equilibrium policy based on the current model theta_t
        print(f"  Solving for equilibrium policy pi_t in M(theta_{t-1})...")
        # Note: At t=1, theta_t is the initial uniform theta_0.
        #       At t>1, theta_t is the model updated in the *previous* iteration (t-1).
        pi1_t, pi2_t, fomwu_gap = solve_ne_fomwu(
            H, S, A1, A2, phi, theta_t, r, K_fomwu_inner, fomwu_eta, fomwu_tau, fomwu_alpha_factor, start_state_dist=start_state_dist
        )
        print(f"  Equilibrium policy found (FOMWU gap for M(theta_t-1): {fomwu_gap:.4e}).")
        # Make sure policies returned by FOMWU are valid
        pi1_t = pi1_t.clamp(min=0); pi1_t /= pi1_t.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        pi2_t = pi2_t.clamp(min=0); pi2_t /= pi2_t.sum(dim=-1, keepdim=True).clamp(min=1e-9)


        # --- Calculate Duality Gap of pi_t in the TRUE GAME M(theta*) ---
        current_duality_gap = float('nan')
        with torch.no_grad():
             pi1_t_no_grad = pi1_t.detach()
             pi2_t_no_grad = pi2_t.detach()
             br_val_p1_true, _ = compute_best_response_value_and_policy(
                 H, S, A1, A2, 0, pi2_t_no_grad, P_star_tensor, r, fomwu_tau, start_state_dist, requires_grad=False
             )
             br_val_p2_true, _ = compute_best_response_value_and_policy(
                 H, S, A1, A2, 1, pi1_t_no_grad, P_star_tensor, r, fomwu_tau, start_state_dist, requires_grad=False
             )
             if not torch.isnan(br_val_p1_true) and not torch.isnan(br_val_p2_true):
                 current_duality_gap = (br_val_p1_true - br_val_p2_true).item()
             else:
                 print(f"  Warning: NaN encountered during true duality gap calculation for t={t}.")
        duality_gaps_history.append(current_duality_gap)
        print(f"  Duality Gap of pi_t in True Game M(theta*): {current_duality_gap:.6f}")
        # --- End Duality Gap Calculation ---


        # --- 2. Update model parameter theta_t (uses pi_t from step 1) ---
        print("  Updating model parameter theta_t...")
        # Note: pi1_t, pi2_t are from FOMWU based on the model from the *start* of this iteration
        theta_t_new = update_model_vmg(
            H, S, A1, A2, d, phi, theta_t, dataset, pi1_t, pi2_t, r, current_vmg_alpha,
            vmg_lr, model_optim_steps, fomwu_tau, start_state_dist
        )
        theta_t = theta_t_new.detach() # theta_t now becomes the model for iteration t+1
        print("  Model parameter updated.")
        thetas_history.append(theta_t.cpu().numpy()) # Store theta *after* update

        theta_error = torch.norm(theta_t - theta_star)
        theta_errors.append(theta_error.item())
        print(f"  Theta Error ||theta_t - theta*||: {theta_error.item():.6f}")


        # --- 3. Data Collection (uses pi_t from step 1 and *updated* theta_t from step 2) ---
        print("  Collecting data...")
        # 3a. Simulate using pi_t in True Game M(theta*)
        initial_s_eq = np.random.choice(S) if S > 0 else 0
        trajectory_eq = simulate_trajectory(H, S, A1, A2, pi1_t, pi2_t, phi, theta_star, start_state=initial_s_eq)
        for h_traj, s_traj, a1_traj, a2_traj, sn_traj in trajectory_eq:
            if 0 <= h_traj < H: dataset[h_traj].append((s_traj, a1_traj, a2_traj, sn_traj))

        # 3b. Compute BR policies pi_tilde_n under NEW model theta_t (using pi_t from step 1)
        with torch.no_grad():
             theta_t_valid = theta_t.clone() # Use the newly updated theta_t
             for h_p in range(H): theta_t_valid[h_p] = project_simplex(theta_t[h_p], z=1.0)
             P_t_tensor = torch.stack([compute_transition(phi[h], theta_t_valid[h]) for h in range(H)])

             _, pi1_tilde_t = compute_best_response_value_and_policy(
                 H, S, A1, A2, 0, pi2_t, P_t_tensor, r, fomwu_tau, start_state_dist, requires_grad=False
             )
             _, pi2_tilde_t = compute_best_response_value_and_policy(
                 H, S, A1, A2, 1, pi1_t, P_t_tensor, r, fomwu_tau, start_state_dist, requires_grad=False
             )
             # Ensure BR policies are valid probabilities
             pi1_tilde_t = pi1_tilde_t.clamp(min=0); pi1_tilde_t /= pi1_tilde_t.sum(dim=-1, keepdim=True).clamp(min=1e-9)
             pi2_tilde_t = pi2_tilde_t.clamp(min=0); pi2_tilde_t /= pi2_tilde_t.sum(dim=-1, keepdim=True).clamp(min=1e-9)


        # 3c. Simulate using BR policies in True Game M(theta*)
        initial_s_p1br = np.random.choice(S) if S > 0 else 0
        trajectory_p1br = simulate_trajectory(H, S, A1, A2, pi1_tilde_t, pi2_t, phi, theta_star, start_state=initial_s_p1br)
        for h_traj, s_traj, a1_traj, a2_traj, sn_traj in trajectory_p1br:
             if 0 <= h_traj < H: dataset[h_traj].append((s_traj, a1_traj, a2_traj, sn_traj))

        initial_s_p2br = np.random.choice(S) if S > 0 else 0
        trajectory_p2br = simulate_trajectory(H, S, A1, A2, pi1_t, pi2_tilde_t, phi, theta_star, start_state=initial_s_p2br)
        for h_traj, s_traj, a1_traj, a2_traj, sn_traj in trajectory_p2br:
             if 0 <= h_traj < H: dataset[h_traj].append((s_traj, a1_traj, a2_traj, sn_traj))

        total_samples_iter = len(trajectory_eq) + len(trajectory_p1br) + len(trajectory_p2br)
        # print(f"  Finished collecting {total_samples_iter} transitions this iteration.")

        iter_time = time.time() - iter_start_time
        print(f"  Iteration {t} complete. Time: {iter_time:.2f}s")

        # Progress Check
        # if t % 10 == 0 or t == T:
        #      total_samples = sum(len(step_data) for step_data in dataset)
        #      print(f"\n==== Progress Check @ Iteration {t}/{T} (Seed: {seed}) ====")
        #      print(f"  Total samples collected: {total_samples}")
        #      print(f"  Current Theta Error ||theta_t - theta*||: {theta_error.item():.6f}")
        #      print(f"  Current Duality Gap (True Game): {current_duality_gap:.6f}")
        #      print(f"========================================\n")


    print(f"\nVMG Finished (Seed: {seed}).")
    final_theta_error = theta_errors[-1]
    final_duality_gap = duality_gaps_history[-1] if duality_gaps_history else float('nan')
    print(f"Final Theta Error ||theta_T - theta*||: {final_theta_error:.6f}")
    print(f"Final Duality Gap (True Game): {final_duality_gap:.6f}")
    total_elapsed = time.time() - start_time
    print(f"Total time for seed {seed}: {total_elapsed:.2f}s")

    # Return theta errors (T+1 points) and duality gaps (T points)
    return thetas_history, theta_errors, duality_gaps_history

# --- Parameters ---
H_horizon = 5
S_states = 5
A1_actions = 5
A2_actions = 5
d_features = 10
T_vmg = 250        # VMG iterations
K_fomwu_inner = 3000 # OMWU iterations
vmg_alpha_base = 0.01 # Weight of value incentive
vmg_lr = 1e-4 # Learning rate for model update (Adam)
model_optim_steps = 100 # Steps for Adam per VMG iteration
fomwu_eta = 0.04 # OMWU policy learning rate
fomwu_tau = 0.05 # Entropy regularization
fomwu_alpha_factor = 1 # Factor for V update LR in OMWU
num_seeds = 3
base_seed = 1
random_seeds = [base_seed + 4*i for i in range(num_seeds)]

# --- Create Results Directory ---
results_dir = "vmg_results"
os.makedirs(results_dir, exist_ok=True)

# --- Define Filenames ---
data_filename_template = "vmg_results_data_H{}_S{}_d{}_T{}_seeds{}.npz"
plot_filename_template = "vmg_avg_metrics_H{}_S{}_d{}_T{}_seeds{}.png"

data_filename = os.path.join(results_dir, data_filename_template.format(
    H_horizon, S_states, d_features, T_vmg, num_seeds))
plot_filename = os.path.join(results_dir, plot_filename_template.format(
    H_horizon, S_states, d_features, T_vmg, num_seeds))

# --- Run Experiments Across Seeds ---
all_theta_errors = []
all_duality_gaps = []

# Check if data file already exists to potentially skip runs
run_experiments = True
if os.path.exists(data_filename):
    print(f"Found existing data file: {data_filename}")
    try:
        # Try loading to see if it's valid and matches parameters
        loaded_data = np.load(data_filename)
        if loaded_data['theta_errors'].shape[0] == num_seeds and \
           loaded_data['theta_errors'].shape[1] == T_vmg + 1 and \
           loaded_data['duality_gaps'].shape[0] == num_seeds and \
           loaded_data['duality_gaps'].shape[1] == T_vmg:
            print("Existing data file matches parameters. Skipping runs.")
            run_experiments = False
        else:
            print("Existing data file parameter mismatch. Re-running experiments.")
    except Exception as e:
        print(f"Error loading existing data file ({e}). Re-running experiments.")

if run_experiments:
    print("Running experiments...")
    for seed in random_seeds:
        _, errors_hist, gaps_hist = run_vmg(
            H_horizon, S_states, A1_actions, A2_actions, d_features, T_vmg, K_fomwu_inner,
            vmg_alpha_base, vmg_lr, model_optim_steps, fomwu_eta, fomwu_tau, fomwu_alpha_factor,
            seed=seed
        )
        all_theta_errors.append(errors_hist)
        all_duality_gaps.append(gaps_hist)

    # Convert lists to numpy arrays for saving
    theta_errors_np = np.array(all_theta_errors)
    duality_gaps_np = np.array(all_duality_gaps) # May contain NaNs

    # --- Save Results ---
    try:
        np.savez(data_filename, theta_errors=theta_errors_np, duality_gaps=duality_gaps_np)
        print(f"\nResults saved to {data_filename}")
    except Exception as e:
        print(f"\nError saving results to {data_filename}: {e}")
else:
    print("Loading results from file...")
    try:
        loaded_data = np.load(data_filename)
        theta_errors_np = loaded_data['theta_errors']
        duality_gaps_np = loaded_data['duality_gaps']
        print(f"Results loaded successfully from {data_filename}")
    except Exception as e:
        print(f"Error loading results from {data_filename}: {e}")
        # Exit or handle error appropriately if loading fails
        exit()

# --- Process Results (Calculate Mean/Std) ---
# Check if data was loaded/generated successfully before processing
if 'theta_errors_np' in locals() and 'duality_gaps_np' in locals():
    mean_theta_errors = np.mean(theta_errors_np, axis=0)
    std_theta_errors = np.std(theta_errors_np, axis=0)

    # Use nanmean/nanstd for gaps to handle potential NaNs gracefully
    mean_duality_gaps = np.nanmean(duality_gaps_np, axis=0)
    std_duality_gaps = np.nanstd(duality_gaps_np, axis=0)

    processing_successful = True
    print("\nMean and standard deviation calculated.")
else:
    processing_successful = False
    print("\nError: Data arrays not available for processing. Cannot plot.")

"""# Plots"""

# --- Plotting with Mean and Variance (Separate Figures) ---
if processing_successful:
    iterations_error = np.arange(T_vmg + 1) # 0 to T
    iterations_gap = np.arange(1, T_vmg + 1) # 1 to T

    # --- Plot 1: Theta Error ---
    plt.figure(figsize=(10, 6)) # Create figure for Theta Error
    plt.plot(iterations_error, mean_theta_errors, marker='o', linestyle='-', markersize=3, label='Mean Theta Error')
    plt.fill_between(iterations_error,
                     mean_theta_errors - std_theta_errors,
                     mean_theta_errors + std_theta_errors,
                     alpha=0.3, label='Mean ± 1 Std Dev')
    plt.ylabel("L2 Error || theta_t - theta* ||")
    plt.title(f"VMG Model Parameter Error (Avg over {num_seeds} seeds)\n"
              f"H={H_horizon}, S={S_states}, A={A1_actions}x{A2_actions}, d={d_features}, T={T_vmg}")
    plt.xlabel("VMG Iteration t")
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    # Define filename and save Theta Error plot
    plot_filename_error = os.path.join(results_dir, f"vmg_avg_theta_error_H{H_horizon}_S{S_states}_d{d_features}_T{T_vmg}_seeds{num_seeds}.png")
    try:
        plt.savefig(plot_filename_error)
        print(f"\nTheta Error plot saved to {plot_filename_error}")
    except Exception as e:
        print(f"\nError saving Theta Error plot to {plot_filename_error}: {e}")
    plt.show() # Display the plot
    plt.close() # Close the figure to free memory


    # --- Plot 2: Duality Gap ---
    plt.figure(figsize=(10, 6)) # Create figure for Duality Gap

    # Filter out potential NaNs in mean/std before plotting fill_between
    valid_gap_indices = ~np.isnan(mean_duality_gaps) & ~np.isnan(std_duality_gaps)
    if np.any(valid_gap_indices):
        plt.plot(iterations_gap[valid_gap_indices], mean_duality_gaps[valid_gap_indices], marker='x', linestyle='--', markersize=4, color='r', label='Mean Duality Gap')
        plt.fill_between(iterations_gap[valid_gap_indices],
                         (mean_duality_gaps - std_duality_gaps)[valid_gap_indices],
                         (mean_duality_gaps + std_duality_gaps)[valid_gap_indices],
                         alpha=0.3, color='r', label='Mean ± 1 Std Dev')
    else:
        print("Warning: Could not plot duality gap variance due to NaNs or empty data.")
        plt.plot([], [], marker='x', linestyle='--', markersize=4, color='r', label='Mean Duality Gap (NaN/Empty)')

    plt.xlabel("VMG Iteration t")
    plt.ylabel("Duality Gap")
    plt.title(f"Duality Gap v.s. Number of Iterations (Avg over {num_seeds} seeds)\n"
              f"H={H_horizon}, S={S_states}, A={A1_actions}x{A2_actions}, d={d_features}, T={T_vmg}")
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    # Define filename and save Duality Gap plot
    plot_filename_gap = os.path.join(results_dir, f"vmg_avg_duality_gap_H{H_horizon}_S{S_states}_d{d_features}_T{T_vmg}_seeds{num_seeds}.png")
    try:
        plt.savefig(plot_filename_gap)
        print(f"Duality Gap plot saved to {plot_filename_gap}")
    except Exception as e:
        print(f"\nError saving Duality Gap plot to {plot_filename_gap}: {e}")
    plt.show() # Display the plot
    plt.close() # Close the figure

    # --- Print Final Metrics ---
    print(f"\nFinal Mean Theta Error: {mean_theta_errors[-1]:.6f} ± {std_theta_errors[-1]:.6f}")
    if len(mean_duality_gaps) > 0 and not np.isnan(mean_duality_gaps[-1]):
        print(f"Final Mean Duality Gap: {mean_duality_gaps[-1]:.6f} ± {std_duality_gaps[-1]:.6f}")
    else:
        print(f"Final Mean Duality Gap: N/A (NaN or empty)")

else:
    print("\nSkipping plotting due to previous errors in data loading/processing.")