import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import warnings
from scipy.special import factorial
warnings.filterwarnings('ignore')


def cognitive_logits(sv_payoff11, sv_payoff12, sv_payoff21, sv_payoff22,
                     fv_payoff11, fv_payoff12, fv_payoff21, fv_payoff22, lambda_, level=6):
    sv_logit_dict = {}
    sv_prob_dict = {}
    fv_logit_dict = {}
    fv_prob_dict = {}

    sv_prob_dict[0] = torch.tensor([[0.5, 0.5]]).to(sv_payoff21.device)
    fv_prob_dict[0] = torch.tensor([[0.5, 0.5]]).to(sv_payoff21.device)

    poisson_pmf = lambda y, mu: mu ** y / factorial(y) * torch.exp(-mu)
    p_level = []

    for i in range(len(lambda_)):
        sub_p = torch.zeros(level)
        sub_lamb = lambda_[i, :]
        sub_level = min(int(sub_lamb), level-1)
        for j in range(sub_level+1):
            sub_p[j] = poisson_pmf(j, sub_lamb)
        sub_p = sub_p / torch.sum(sub_p)
        p_level.append(sub_p.unsqueeze(0))
    p_level = torch.cat(p_level).to(sv_payoff11.device)

    each_level_poisson = []
    for i in range(1, level+1):
        sub_poisson = torch.tensor([poisson_pmf(j, torch.tensor(i)) for j in range(i)])
        sub_poisson = sub_poisson / torch.sum(sub_poisson)
        each_level_poisson.append(sub_poisson.to(sv_payoff21.device))

    for i in range(1, level+1):
        sv_logit_dict[i] = []
        fv_logit_dict[i] = []
        sub_c1_logits_sv = 0
        sub_c2_logits_sv = 0
        sub_c1_logits_fv = 0
        sub_c2_logits_fv = 0
        for j in range(i):
            sub_c1_logits_sv += each_level_poisson[i-1][j] * (fv_prob_dict[j][:, 0].unsqueeze(1)*sv_payoff11 + fv_prob_dict[j][:, 1].unsqueeze(1)*sv_payoff12)
            sub_c2_logits_sv += each_level_poisson[i-1][j] * (fv_prob_dict[j][:, 0].unsqueeze(1)*sv_payoff21 + fv_prob_dict[j][:, 1].unsqueeze(1)*sv_payoff22)

            sub_c1_logits_fv += each_level_poisson[i-1][j] * (sv_prob_dict[j][:, 0].unsqueeze(1) * fv_payoff11 + sv_prob_dict[j][:, 1].unsqueeze(1) * fv_payoff12)
            sub_c2_logits_fv += each_level_poisson[i-1][j] * (sv_prob_dict[j][:, 0].unsqueeze(1) * fv_payoff21 + sv_prob_dict[j][:, 1].unsqueeze(1) * fv_payoff22)

        sub_logit_sv = torch.cat([sub_c1_logits_sv, sub_c2_logits_sv], dim=1)
        sub_probs_sv = torch.softmax(sub_logit_sv, dim=-1)

        sub_logit_fv = torch.cat([sub_c1_logits_fv, sub_c2_logits_fv], dim=1)
        sub_probs_fv = torch.softmax(sub_logit_fv, dim=-1)

        sv_logit_dict[i].append(sub_c1_logits_sv)
        sv_logit_dict[i].append(sub_c2_logits_sv)
        sv_prob_dict[i] = sub_probs_sv

        fv_logit_dict[i].append(sub_c1_logits_fv)
        fv_logit_dict[i].append(sub_c2_logits_fv)
        fv_prob_dict[i] = sub_probs_fv

    sv_logit_c1 = torch.sum(p_level * torch.cat([sv_logit_dict[i][0] for i in range(1, level+1)], dim=1), dim=1).unsqueeze(1)
    sv_logit_c2 = torch.sum(p_level * torch.cat([sv_logit_dict[i][1] for i in range(1, level+1)], dim=1), dim=1).unsqueeze(1)
    fv_logit_c1 = torch.sum(p_level * torch.cat([fv_logit_dict[i][0] for i in range(1, level+1)], dim=1), dim=1).unsqueeze(1)
    fv_logit_c2 = torch.sum(p_level * torch.cat([fv_logit_dict[i][1] for i in range(1, level+1)], dim=1), dim=1).unsqueeze(1)

    final_sv_logits = torch.cat([sv_logit_c1, sv_logit_c2], dim=1)
    final_fv_logits = torch.cat([fv_logit_c1, fv_logit_c2], dim=1)

    return final_sv_logits, final_fv_logits


class PayoffNet(nn.Module):
    def __init__(self, hid_dim=16):
        super(PayoffNet, self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(16, hid_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hid_dim, hid_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hid_dim, 4))

    def forward(self, x):
        x = self.mlp(x)

        return x


class Emitter(nn.Module):
    def __init__(self, hid_dim=16):
        super().__init__()
        self.sv_l = nn.Sequential(nn.Linear(12, hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, 2))
        self.fv_l = nn.Sequential(nn.Linear(12, hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, 2))
        self.ad_sv = nn.Linear(14, 1)
        self.ad_fv = nn.Linear(14, 1)

        self.relu = nn.ReLU()

    def forward(self, l_t, o_t, sv_c11, sv_c12, sv_c21, sv_c22,
                                fv_c11, fv_c12, fv_c21, fv_c22):
        logits_sv, logits_fv = cognitive_logits(sv_c11, sv_c12, sv_c21, sv_c22,
                                                fv_c11, fv_c12, fv_c21, fv_c22,
                                                l_t, level=5)

        ps_sv = torch.softmax(logits_sv, dim=1)[:, 1].unsqueeze(-1)
        ps_fv = torch.softmax(logits_fv, dim=1)[:, 1].unsqueeze(-1)

        return ps_sv, ps_fv


class GatedTransitionZ(nn.Module):
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_b_to_hidden = nn.Linear(2, transition_dim)

        self.lin_gate_hidden_to_all = nn.Linear(transition_dim, z_dim)

        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_b_to_hidden = nn.Linear(2, transition_dim)

        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_hidden_to_b = nn.Linear(transition_dim, z_dim)

        self.lin_sig_z = nn.Linear(z_dim, z_dim)
        self.lin_sig_b = nn.Linear(z_dim, z_dim)

        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_b_to_loc = nn.Linear(z_dim, z_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, combine_z):
        z_t_1 = combine_z[:, 0:1]
        behavior = combine_z[:, 1:]
        _gate = (self.relu(self.lin_gate_b_to_hidden(behavior)) + self.relu(self.lin_gate_z_to_hidden(z_t_1)))*0.5
        gate = torch.sigmoid(self.lin_gate_hidden_to_all(_gate))

        _proposed_mean_b = self.relu(self.lin_proposed_mean_b_to_hidden(behavior))
        proposed_mean_b = self.lin_proposed_mean_hidden_to_b(_proposed_mean_b)

        _proposed_mean_z = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean_z = self.lin_proposed_mean_hidden_to_z(_proposed_mean_z)

        loc = (proposed_mean_z) * (1-gate) + (proposed_mean_b) * gate

        scale = self.softplus(self.lin_sig_z(self.relu(proposed_mean_z))) * (1 - gate)
        scale += self.softplus(self.lin_sig_b(self.relu(proposed_mean_b))) * gate

        return loc, scale


class GatedTransitionL(nn.Module):
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_b_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_all = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig_z = nn.Linear(z_dim, z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, combine_z):
        z_t_1 = combine_z[:, 0:1]
        behavior = combine_z[:, 1:2]

        _gate = self.relu(self.lin_gate_b_to_hidden(behavior))
        gate = torch.sigmoid(self.lin_gate_hidden_to_all(_gate))

        _proposed_mean_z = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean_z = self.lin_proposed_mean_hidden_to_z(_proposed_mean_z)

        loc = self.softplus(proposed_mean_z) * gate + z_t_1 * (1 - gate)
        scale = self.softplus(self.lin_sig_z(self.relu(proposed_mean_z)))

        return loc, scale


class GatedTransitionPayoff(nn.Module):
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_b_to_hidden = nn.Linear(14, transition_dim)

        self.lin_gate_hidden_to_6z = nn.Linear(transition_dim, z_dim)
        self.lin_gate_hidden_to_b = nn.Linear(transition_dim, z_dim)
        self.lin_gate_hidden_to_all = nn.Linear(transition_dim, z_dim)

        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_b_to_hidden = nn.Linear(14, transition_dim)

        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_hidden_to_b = nn.Linear(transition_dim, z_dim)

        self.lin_sig_z = nn.Linear(z_dim, z_dim)
        self.lin_sig_b = nn.Linear(z_dim, z_dim)

        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_b_to_loc = nn.Linear(z_dim, z_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, combined_p):
        obs = combined_p[:, :14]
        prev = combined_p[:, 14:]
        _gate = (self.relu(self.lin_gate_b_to_hidden(obs)) + self.relu(self.lin_gate_z_to_hidden(prev)))*0.5
        gate = torch.sigmoid(self.lin_gate_hidden_to_all(_gate))

        _proposed_mean_b = self.relu(self.lin_proposed_mean_b_to_hidden(obs))
        proposed_mean_b = self.lin_proposed_mean_hidden_to_b(_proposed_mean_b)

        _proposed_mean_z = self.relu(self.lin_proposed_mean_z_to_hidden(prev))
        proposed_mean_z = self.lin_proposed_mean_hidden_to_z(_proposed_mean_z)

        loc = (proposed_mean_z) * (1-gate) + (proposed_mean_b) * gate

        scale = self.softplus(self.lin_sig_z(self.relu(proposed_mean_z))) * (1 - gate)
        scale += self.softplus(self.lin_sig_b(self.relu(proposed_mean_b))) * gate

        # for simplicity, we treat scale = 0
        return loc


class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(2, rnn_dim)
        self.lin_b_to_hidden = nn.Linear(2, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        self.tanh = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, b_t_1, h_rnn):
        h_combined = 0.33 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + self.tanh(self.lin_b_to_hidden(b_t_1)) + h_rnn)

        loc = self.softplus(self.lin_hidden_to_loc(h_combined))
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))

        return loc, scale
