from model.module import *
import warnings
warnings.filterwarnings('ignore')


class DMCHM(nn.Module):
    def __init__(self, input_dim=15, h_dim=1, transition_dim=16, rnn_dim=16, num_layers=1):
        super().__init__()
        # payoff network
        self.sv_payoff = PayoffNet()
        self.fv_payoff = PayoffNet()

        # sv modules
        self.emitter_sv = Emitter()
        self.trans_z_sv = GatedTransitionZ(h_dim, transition_dim)
        self.trans_l_sv = GatedTransitionL(h_dim, transition_dim)
        self.com_z_sv = Combiner(h_dim, rnn_dim * 2)
        self.com_l_sv = Combiner(h_dim, rnn_dim * 2)

        self.sv1 = GatedTransitionPayoff(h_dim, transition_dim)
        self.sv2 = GatedTransitionPayoff(h_dim, transition_dim)
        self.sv3 = GatedTransitionPayoff(h_dim, transition_dim)
        self.sv4 = GatedTransitionPayoff(h_dim, transition_dim)

        self.rnn_z_sv = nn.GRU(input_size=input_dim, hidden_size=rnn_dim,
                               batch_first=True, bidirectional=True, num_layers=num_layers)
        self.rnn_l_sv = nn.GRU(input_size=input_dim, hidden_size=rnn_dim,
                               batch_first=True, bidirectional=True, num_layers=num_layers)
        self.z_0_sv = nn.Parameter(torch.abs(torch.randn(h_dim)))
        self.l_0_sv = nn.Parameter(torch.abs(torch.randn(h_dim))+1)

        # fv modules
        self.emitter_fv = Emitter()
        self.trans_z_fv = GatedTransitionZ(h_dim, transition_dim)
        self.trans_l_fv = GatedTransitionL(h_dim, transition_dim)
        self.com_z_fv = Combiner(h_dim, rnn_dim * 2)
        self.com_l_fv = Combiner(h_dim, rnn_dim * 2)

        self.fv1 = GatedTransitionPayoff(h_dim, transition_dim)
        self.fv2 = GatedTransitionPayoff(h_dim, transition_dim)
        self.fv3 = GatedTransitionPayoff(h_dim, transition_dim)
        self.fv4 = GatedTransitionPayoff(h_dim, transition_dim)

        self.rnn_z_fv = nn.GRU(input_size=input_dim, hidden_size=rnn_dim,
                               batch_first=True, bidirectional=True, num_layers=num_layers)
        self.rnn_l_fv = nn.GRU(input_size=input_dim, hidden_size=rnn_dim,
                               batch_first=True, bidirectional=True, num_layers=num_layers)
        self.z_0_fv = nn.Parameter(torch.abs(torch.randn(h_dim)))
        self.l_0_fv = nn.Parameter(torch.abs(torch.randn(h_dim))+1)

        # payoffs
        self.p_0_sv1 = nn.Parameter(torch.randn(1))
        self.p_0_sv2 = nn.Parameter(torch.randn(1))
        self.p_0_sv3 = nn.Parameter(torch.randn(1))
        self.p_0_sv4 = nn.Parameter(torch.randn(1))
        self.p_0_fv1 = nn.Parameter(torch.randn(1))
        self.p_0_fv2 = nn.Parameter(torch.randn(1))
        self.p_0_fv3 = nn.Parameter(torch.randn(1))
        self.p_0_fv4 = nn.Parameter(torch.randn(1))

    def prior(self, T_max, sv_sequences, fv_sequences, agg_sv, agg_fv):
        p_z_loc_sv, p_z_scale_sv, p_z_loc_fv, p_z_scale_fv = [], [], [], []
        p_l_loc_sv, p_l_scale_sv, p_l_loc_fv, p_l_scale_fv = [], [], [], []

        z_prev_sv = self.z_0_sv.expand(sv_sequences.size(0), self.z_0_sv.size(0))
        l_prev_sv = self.l_0_sv.expand(sv_sequences.size(0), self.l_0_sv.size(0))
        z_prev_fv = self.z_0_fv.expand(fv_sequences.size(0), self.z_0_fv.size(0))
        l_prev_fv = self.l_0_fv.expand(fv_sequences.size(0), self.l_0_fv.size(0))

        for t in range(1, T_max + 1):
            behavior_t_sv = agg_sv[:, t - 1, :]
            behavior_t_fv = agg_fv[:, t - 1, :]

            # z
            combine_z_sv = torch.cat([z_prev_sv, behavior_t_sv, behavior_t_fv], dim=1)
            combine_z_fv = torch.cat([z_prev_fv, behavior_t_sv, behavior_t_fv], dim=1)
            z_loc_sv, z_scale_sv = self.trans_z_sv(combine_z_sv)
            z_loc_fv, z_scale_fv = self.trans_z_fv(combine_z_fv)
            epsilon1 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            epsilon2 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            z_t_sv = torch.abs(z_loc_sv + epsilon1 * z_scale_sv)
            z_t_fv = torch.abs(z_loc_fv + epsilon2 * z_scale_fv)

            # lambda
            combine_l_sv = torch.cat([l_prev_sv, z_t_sv], dim=1)
            combine_l_fv = torch.cat([l_prev_fv, z_t_fv], dim=1)
            l_loc_sv, l_scale_sv = self.trans_l_sv(combine_l_sv)
            l_loc_fv, l_scale_fv = self.trans_l_fv(combine_l_fv)
            epsilon3 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            epsilon4 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            l_t_sv = torch.abs(l_loc_sv + epsilon3 * l_scale_sv) + 1
            l_t_fv = torch.abs(l_loc_fv + epsilon4 * l_scale_fv) + 1
            l_t_sv[torch.where(l_t_sv > 5)[0]] = 5
            l_t_fv[torch.where(l_t_fv > 5)[0]] = 5

            # store res
            p_z_loc_sv.append(z_loc_sv.unsqueeze(-1))
            p_z_loc_fv.append(z_loc_fv.unsqueeze(-1))
            p_l_loc_sv.append(l_loc_sv.unsqueeze(-1))
            p_l_loc_fv.append(l_loc_fv.unsqueeze(-1))

            p_z_scale_sv.append(z_scale_sv.unsqueeze(-1))
            p_z_scale_fv.append(z_scale_fv.unsqueeze(-1))
            p_l_scale_sv.append(l_scale_sv.unsqueeze(-1))
            p_l_scale_fv.append(l_scale_fv.unsqueeze(-1))

            z_prev_sv = z_t_sv
            l_prev_sv = l_t_sv
            z_prev_fv = z_t_fv
            l_prev_fv = l_t_fv

        p_z_loc_sv = torch.cat(p_z_loc_sv, dim=1)
        p_z_loc_fv = torch.cat(p_z_loc_fv, dim=1)
        p_l_loc_sv = torch.cat(p_l_loc_sv, dim=1)
        p_l_loc_fv = torch.cat(p_l_loc_fv, dim=1)

        p_z_scale_sv = torch.cat(p_z_scale_sv, dim=1)
        p_z_scale_fv = torch.cat(p_z_scale_fv, dim=1)
        p_l_scale_sv = torch.cat(p_l_scale_sv, dim=1)
        p_l_scale_fv = torch.cat(p_l_scale_fv, dim=1)

        return p_z_loc_sv, p_z_scale_sv, p_z_loc_fv, p_z_scale_fv, \
               p_l_loc_sv, p_l_scale_sv, p_l_loc_fv, p_l_scale_fv

    def posterior(self, T_max, sv_sequences, fv_sequences, obs, agg_sv, agg_fv):
        # initialization
        z_prev_sv = self.z_0_sv.expand(sv_sequences.size(0), self.z_0_sv.size(0))
        l_prev_sv = self.l_0_sv.expand(sv_sequences.size(0), self.l_0_sv.size(0))
        z_prev_fv = self.z_0_fv.expand(fv_sequences.size(0), self.z_0_fv.size(0))
        l_prev_fv = self.l_0_fv.expand(fv_sequences.size(0), self.l_0_fv.size(0))

        sv1_prev = self.p_0_sv1.expand(sv_sequences.size(0), 1).to(obs.device)
        sv2_prev = self.p_0_sv2.expand(sv_sequences.size(0), 1).to(obs.device)
        sv3_prev = self.p_0_sv3.expand(sv_sequences.size(0), 1).to(obs.device)
        sv4_prev = self.p_0_sv4.expand(sv_sequences.size(0), 1).to(obs.device)
        fv1_prev = self.p_0_fv1.expand(fv_sequences.size(0), 1).to(obs.device)
        fv2_prev = self.p_0_fv2.expand(fv_sequences.size(0), 1).to(obs.device)
        fv3_prev = self.p_0_fv3.expand(fv_sequences.size(0), 1).to(obs.device)
        fv4_prev = self.p_0_fv4.expand(fv_sequences.size(0), 1).to(obs.device)

        # inference
        rnn_z_sv, _ = self.rnn_z_sv(torch.cat([obs, sv_sequences], dim=-1))
        rnn_z_fv, _ = self.rnn_z_fv(torch.cat([obs, fv_sequences], dim=-1))
        rnn_l_sv, _ = self.rnn_l_sv(torch.cat([obs, sv_sequences], dim=-1))
        rnn_l_fv, _ = self.rnn_l_fv(torch.cat([obs, fv_sequences], dim=-1))

        q_z_loc_sv, q_z_scale_sv, q_z_loc_fv, q_z_scale_fv = [], [], [], []
        q_l_loc_sv, q_l_scale_sv, q_l_loc_fv, q_l_scale_fv = [], [], [], []
        sv_preds, fv_preds = [], []

        for t in range(1, T_max + 1):
            behavior_t_sv = agg_sv[:, t - 1, :]
            behavior_t_fv = agg_fv[:, t - 1, :]
            behavior_t = torch.cat([behavior_t_sv, behavior_t_fv], dim=1)
            z_prev = torch.cat([z_prev_sv, z_prev_fv], dim=1)
            z_loc_sv, z_scale_sv = self.com_z_sv(z_prev, behavior_t, rnn_z_sv[:, t - 1, :])
            z_loc_fv, z_scale_fv = self.com_z_fv(z_prev, behavior_t, rnn_z_fv[:, t - 1, :])
            epsilon1 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            epsilon2 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            z_t_sv = torch.abs(z_loc_sv + epsilon1 * z_scale_sv)
            z_t_fv = torch.abs(z_loc_fv + epsilon2 * z_scale_fv)

            z_t = torch.cat([z_t_sv, z_t_fv], dim=1)
            l_prev = torch.cat([l_prev_sv, l_prev_fv], dim=1)
            l_loc_sv, l_scale_sv = self.com_l_sv(l_prev, z_t, rnn_l_sv[:, t - 1, :])
            l_loc_fv, l_scale_fv = self.com_l_fv(l_prev, z_t, rnn_l_fv[:, t - 1, :])

            epsilon3 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            epsilon4 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            l_t_sv = torch.abs(l_loc_sv + epsilon3 * l_scale_sv) + 1
            l_t_fv = torch.abs(l_loc_fv + epsilon4 * l_scale_fv) + 1
            l_t_sv[torch.where(l_t_sv > 5)[0]] = 5
            l_t_fv[torch.where(l_t_fv > 5)[0]] = 5

            q_z_loc_sv.append(z_loc_sv.unsqueeze(-1))
            q_z_loc_fv.append(z_loc_fv.unsqueeze(-1))
            q_l_loc_sv.append(l_loc_sv.unsqueeze(-1))
            q_l_loc_fv.append(l_loc_fv.unsqueeze(-1))

            q_z_scale_sv.append(z_scale_sv.unsqueeze(-1))
            q_z_scale_fv.append(z_scale_fv.unsqueeze(-1))
            q_l_scale_sv.append(l_scale_sv.unsqueeze(-1))
            q_l_scale_fv.append(l_scale_fv.unsqueeze(-1))

            # payoffs
            sv1 = self.sv1(torch.cat([obs[:, t - 1, :], sv1_prev], dim=1))
            sv2 = self.sv2(torch.cat([obs[:, t - 1, :], sv2_prev], dim=1))
            sv3 = self.sv3(torch.cat([obs[:, t - 1, :], sv3_prev], dim=1))
            sv4 = self.sv4(torch.cat([obs[:, t - 1, :], sv4_prev], dim=1))
            fv1 = self.fv1(torch.cat([obs[:, t - 1, :], fv1_prev], dim=1))
            fv2 = self.fv2(torch.cat([obs[:, t - 1, :], fv2_prev], dim=1))
            fv3 = self.fv3(torch.cat([obs[:, t - 1, :], fv3_prev], dim=1))
            fv4 = self.fv4(torch.cat([obs[:, t - 1, :], fv4_prev], dim=1))

            emission_probs_t_sv, _ = self.emitter_sv(l_t_sv, obs[:, t - 1, :],
                                                     sv1, sv2, sv3, sv4,
                                                     fv1, fv2, fv3, fv4)
            _, emission_probs_t_fv = self.emitter_fv(l_t_fv, obs[:, t - 1, :],
                                                     sv1, sv2, sv3, sv4,
                                                     fv1, fv2, fv3, fv4)

            sv_preds.append(emission_probs_t_sv.unsqueeze(-1))
            fv_preds.append(emission_probs_t_fv.unsqueeze(-1))

            z_prev_sv = z_t_sv
            l_prev_sv = l_t_sv
            z_prev_fv = z_t_fv
            l_prev_fv = l_t_fv

            sv1_prev = sv1
            sv2_prev = sv2
            sv3_prev = sv3
            sv4_prev = sv4

            fv1_prev = fv1
            fv2_prev = fv2
            fv3_prev = fv3
            fv4_prev = fv4

        sv_preds = torch.cat(sv_preds, dim=1)
        fv_preds = torch.cat(fv_preds, dim=1)

        q_z_loc_sv = torch.cat(q_z_loc_sv, dim=1)
        q_z_loc_fv = torch.cat(q_z_loc_fv, dim=1)
        q_l_loc_sv = torch.cat(q_l_loc_sv, dim=1)
        q_l_loc_fv = torch.cat(q_l_loc_fv, dim=1)

        q_z_scale_sv = torch.cat(q_z_scale_sv, dim=1)
        q_z_scale_fv = torch.cat(q_z_scale_fv, dim=1)
        q_l_scale_sv = torch.cat(q_l_scale_sv, dim=1)
        q_l_scale_fv = torch.cat(q_l_scale_fv, dim=1)

        return q_z_loc_sv, q_z_scale_sv, q_z_loc_fv, q_z_scale_fv, \
               q_l_loc_sv, q_l_scale_sv, q_l_loc_fv, q_l_scale_fv, sv_preds, fv_preds

    def forward(self, sv_sequences, fv_sequences, obs, agg_sv, agg_fv, mask, weights):
        T_max = sv_sequences.size(1)

        # prior dist
        p_z_loc_sv, p_z_scale_sv, p_z_loc_fv, p_z_scale_fv, \
        p_l_loc_sv, p_l_scale_sv, p_l_loc_fv, p_l_scale_fv = self.prior(T_max, sv_sequences, fv_sequences,
                                                                        agg_sv, agg_fv)

        # posterior dist
        q_z_loc_sv, q_z_scale_sv, q_z_loc_fv, q_z_scale_fv, \
        q_l_loc_sv, q_l_scale_sv, q_l_loc_fv, q_l_scale_fv, \
        sv_preds, fv_preds = self.posterior(T_max, sv_sequences, fv_sequences, obs, agg_sv, agg_fv)

        # kl divergence
        kl_z_sv = self.kl_div(q_z_scale_sv, p_z_scale_sv, q_z_loc_sv, p_z_loc_sv) * mask * weights
        kl_z_fv = self.kl_div(q_z_scale_fv, p_z_scale_fv, q_z_loc_fv, p_z_loc_fv) * mask
        kl_l_sv = self.kl_div(q_l_scale_sv, p_l_scale_sv, q_l_loc_sv, p_l_loc_sv) * mask * weights
        kl_l_fv = self.kl_div(q_l_scale_fv, p_l_scale_fv, q_l_loc_fv, p_l_loc_fv) * mask

        kl_loss = torch.mean(torch.cat([kl_z_sv, kl_z_fv, kl_l_sv, kl_l_fv]))

        return sv_preds, fv_preds, kl_loss

    def kl_div(self, sigma1, sigma2, mu1, mu2):
        kl = torch.log(sigma2 / sigma1) + (sigma1**2+(mu1-mu2)**2) / (2*sigma2**2) - 1/2

        return kl

    def validation(self, sv_sequences, fv_sequences, obs, agg_sv, agg_fv):
        T_max = sv_sequences.size(1)

        z_prev_sv = self.z_0_sv.expand(sv_sequences.size(0), self.z_0_sv.size(0))
        l_prev_sv = self.l_0_sv.expand(sv_sequences.size(0), self.l_0_sv.size(0))
        z_prev_fv = self.z_0_fv.expand(fv_sequences.size(0), self.z_0_fv.size(0))
        l_prev_fv = self.l_0_fv.expand(fv_sequences.size(0), self.l_0_fv.size(0))

        sv1_prev = self.p_0_sv1.expand(sv_sequences.size(0), 1).to(obs.device)
        sv2_prev = self.p_0_sv2.expand(sv_sequences.size(0), 1).to(obs.device)
        sv3_prev = self.p_0_sv3.expand(sv_sequences.size(0), 1).to(obs.device)
        sv4_prev = self.p_0_sv4.expand(sv_sequences.size(0), 1).to(obs.device)

        fv1_prev = self.p_0_fv1.expand(fv_sequences.size(0), 1).to(obs.device)
        fv2_prev = self.p_0_fv2.expand(fv_sequences.size(0), 1).to(obs.device)
        fv3_prev = self.p_0_fv3.expand(fv_sequences.size(0), 1).to(obs.device)
        fv4_prev = self.p_0_fv4.expand(fv_sequences.size(0), 1).to(obs.device)

        rnn_z_sv, _ = self.rnn_z_sv(torch.cat([obs, sv_sequences], dim=-1))
        rnn_z_fv, _ = self.rnn_z_fv(torch.cat([obs, fv_sequences], dim=-1))
        rnn_l_sv, _ = self.rnn_l_sv(torch.cat([obs, sv_sequences], dim=-1))
        rnn_l_fv, _ = self.rnn_l_fv(torch.cat([obs, fv_sequences], dim=-1))

        sv_preds = []
        fv_preds = []

        for t in range(1, T_max + 1):
            behavior_t_sv = agg_sv[:, t - 1, :]
            behavior_t_fv = agg_fv[:, t - 1, :]
            combine_z_sv = torch.cat([z_prev_sv, behavior_t_sv, behavior_t_fv], dim=1)
            combine_z_fv = torch.cat([z_prev_fv, behavior_t_sv, behavior_t_fv], dim=1)
            z_loc_sv, z_scale_sv = self.trans_z_sv(combine_z_sv)
            z_loc_fv, z_scale_fv = self.trans_z_fv(combine_z_fv)
            epsilon1 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            epsilon2 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            z_t_sv = torch.abs(z_loc_sv + epsilon1 * z_scale_sv)
            z_t_fv = torch.abs(z_loc_fv + epsilon2 * z_scale_fv)

            combine_l_sv = torch.cat([l_prev_sv, z_t_sv], dim=1)
            combine_l_fv = torch.cat([l_prev_fv, z_t_fv], dim=1)
            l_loc_sv, l_scale_sv = self.trans_l_sv(combine_l_sv)
            l_loc_fv, l_scale_fv = self.trans_l_fv(combine_l_fv)

            # reparameterized trick
            epsilon3 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            epsilon4 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
            l_t_sv = torch.abs(l_loc_sv + epsilon3 * l_scale_sv) + 1
            l_t_fv = torch.abs(l_loc_fv + epsilon4 * l_scale_fv) + 1
            l_t_sv[torch.where(l_t_sv > 5)[0]] = 5
            l_t_fv[torch.where(l_t_fv > 5)[0]] = 5

            # payoffs
            sv1 = self.sv1(torch.cat([obs[:, t - 1, :], sv1_prev.to(obs.device)], dim=1))
            sv2 = self.sv2(torch.cat([obs[:, t - 1, :], sv2_prev.to(obs.device)], dim=1))
            sv3 = self.sv3(torch.cat([obs[:, t - 1, :], sv3_prev.to(obs.device)], dim=1))
            sv4 = self.sv4(torch.cat([obs[:, t - 1, :], sv4_prev.to(obs.device)], dim=1))
            fv1 = self.fv1(torch.cat([obs[:, t - 1, :], fv1_prev.to(obs.device)], dim=1))
            fv2 = self.fv2(torch.cat([obs[:, t - 1, :], fv2_prev.to(obs.device)], dim=1))
            fv3 = self.fv3(torch.cat([obs[:, t - 1, :], fv3_prev.to(obs.device)], dim=1))
            fv4 = self.fv4(torch.cat([obs[:, t - 1, :], fv4_prev.to(obs.device)], dim=1))

            ##############################################################
            emission_probs_t_sv, _ = self.emitter_sv(l_t_sv, obs[:, t - 1, :],
                                                     sv1, sv2, sv3, sv4,
                                                     fv1, fv2, fv3, fv4)
            _, emission_probs_t_fv = self.emitter_fv(l_t_fv, obs[:, t - 1, :],
                                                     sv1, sv2, sv3, sv4,
                                                     fv1, fv2, fv3, fv4)
            sv_preds.append(emission_probs_t_sv.unsqueeze(-1))
            fv_preds.append(emission_probs_t_fv.unsqueeze(-1))

            z_prev_sv = z_t_sv
            l_prev_sv = l_t_sv
            z_prev_fv = z_t_fv
            l_prev_fv = l_t_fv

            sv1_prev = sv1
            sv2_prev = sv2
            sv3_prev = sv3
            sv4_prev = sv4

            fv1_prev = fv1
            fv2_prev = fv2
            fv3_prev = fv3
            fv4_prev = fv4

        sv_preds = torch.cat(sv_preds, dim=1)
        fv_preds = torch.cat(fv_preds, dim=1)

        return sv_preds, fv_preds

    def stepwise_val(self, z_prev_sv, z_prev_fv, behavior_t_sv, behavior_t_fv, l_prev_sv, l_prev_fv,
                     sv1_prev, sv2_prev, sv3_prev, sv4_prev, fv1_prev, fv2_prev, fv3_prev, fv4_prev, cur_obs):
        combine_z_sv = torch.cat([z_prev_sv, behavior_t_sv, behavior_t_fv], dim=1)
        combine_z_fv = torch.cat([z_prev_fv, behavior_t_sv, behavior_t_fv], dim=1)
        z_loc_sv, z_scale_sv = self.trans_z_sv(combine_z_sv)
        z_loc_fv, z_scale_fv = self.trans_z_fv(combine_z_fv)
        epsilon1 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
        epsilon2 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
        z_t_sv = torch.abs(z_loc_sv + epsilon1 * z_scale_sv)
        z_t_fv = torch.abs(z_loc_fv + epsilon2 * z_scale_fv)

        combine_l_sv = torch.cat([l_prev_sv, z_t_sv], dim=1)
        combine_l_fv = torch.cat([l_prev_fv, z_t_fv], dim=1)
        l_loc_sv, l_scale_sv = self.trans_l_sv(combine_l_sv)
        l_loc_fv, l_scale_fv = self.trans_l_fv(combine_l_fv)

        # reparameterized trick
        epsilon3 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
        epsilon4 = torch.randn(z_loc_sv.shape).to(z_loc_sv.device)
        l_t_sv = torch.abs(l_loc_sv + epsilon3 * l_scale_sv) + 1
        l_t_fv = torch.abs(l_loc_fv + epsilon4 * l_scale_fv) + 1
        l_t_sv[torch.where(l_t_sv > 5)[0]] = 5
        l_t_fv[torch.where(l_t_fv > 5)[0]] = 5

        # payoffs
        sv1 = self.sv1(torch.cat([cur_obs, sv1_prev], dim=1))
        sv2 = self.sv2(torch.cat([cur_obs, sv2_prev], dim=1))
        sv3 = self.sv3(torch.cat([cur_obs, sv3_prev], dim=1))
        sv4 = self.sv4(torch.cat([cur_obs, sv4_prev], dim=1))
        fv1 = self.fv1(torch.cat([cur_obs, fv1_prev], dim=1))
        fv2 = self.fv2(torch.cat([cur_obs, fv2_prev], dim=1))
        fv3 = self.fv3(torch.cat([cur_obs, fv3_prev], dim=1))
        fv4 = self.fv4(torch.cat([cur_obs, fv4_prev], dim=1))

        ##############################################################
        emission_probs_t_sv, _ = self.emitter_sv(l_t_sv, cur_obs,
                                                 sv1, sv2, sv3, sv4,
                                                 fv1, fv2, fv3, fv4)
        _, emission_probs_t_fv = self.emitter_fv(l_t_fv, cur_obs,
                                                 sv1, sv2, sv3, sv4,
                                                 fv1, fv2, fv3, fv4)

        return z_t_sv, z_t_fv, l_t_sv, l_t_fv, \
               sv1, sv2, sv3, sv4, fv1, fv2, fv3, fv4, \
               emission_probs_t_sv, emission_probs_t_fv
