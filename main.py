import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import warnings
from utils.loader import *
from model.dmchm import DMCHM

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test(best_model, test_ds, device):
    best_model.eval()
    sv_sequences_test = test_ds[0].to(device)
    fv_sequences_test = test_ds[1].to(device)
    lengths_test = test_ds[2].to(device)
    obs_test = test_ds[3].to(device)
    agg_sv_test = test_ds[4].to(device)
    agg_fv_test = test_ds[5].to(device)
    loc = test_ds[6].to(device)
    mask_test = test_ds[7].to(device)

    weights_sv = torch.ones(mask_test.shape).to(device)
    weights_fv = torch.ones(mask_test.shape).to(device)
    for i in range(len(lengths_test)):
        weights_sv[i, lengths_test[i] - 1, :] = 10

        fv_seq = fv_sequences_test[i, :lengths_test[i]]
        dec_idx = torch.where(fv_seq == 0)[0]
        weights_fv[i, dec_idx, :] = 10

    # Test
    test_pred_sv, test_pred_fv = best_model.validation(sv_sequences_test, fv_sequences_test, obs_test,
                                                       agg_sv_test, agg_fv_test)

    loss = torch.mean(F.binary_cross_entropy(test_pred_sv, sv_sequences_test, reduction='none', weight=weights_sv) * mask_test) + \
           torch.mean(F.binary_cross_entropy(test_pred_fv, fv_sequences_test, reduction='none', weight=weights_fv) * mask_test)

    discrete_idx_lc = torch.where(((lengths_test > 0) & (lengths_test <= 10)))[0]

    test_pred = test_pred_sv[discrete_idx_lc]
    lengths_test = lengths_test[discrete_idx_lc]
    loc = loc[discrete_idx_lc]
    time_errors = []
    loc_errors = []
    for i in range(test_pred.shape[0]):
        pt = torch.where(test_pred[i, :, :] >= 0.5)[0]
        if len(pt) != 0 and pt[0] <= lengths_test[i] - 1:
            time_error = (lengths_test[i] - 1 - pt[0])
            time_errors.append(time_error.item() * 2)

            loc_error = loc[i, lengths_test[i] - 1] - loc[i, pt[0]]
            loc_errors.append(loc_error.item())

    test_seq_pred = []
    for i in range(len(test_pred)):
        if len(torch.where(test_pred[i, :lengths_test[i], :] >= 0.5)[0]) > 0:
            test_seq_pred.append(1)
        else:
            test_seq_pred.append(0)

    test_label = torch.ones(len(sv_sequences_test))
    acc = accuracy_score(test_label, test_seq_pred)
    print('Test accuracy score {:.4f}, time error {:.4f}, loc error {:.4f}'.format(acc, np.mean(time_errors),
                                                                                   np.mean(loc_errors)))
    best_model.train()

    return loss.item()


def train(dmm, optimizer, scheduler, train_ds, test_ds, weights_sv, weights_fv, device):
    # train set
    sv_sequences = train_ds[0].to(device)
    fv_sequences = train_ds[1].to(device)
    lengths = train_ds[2].to(device)
    obs = train_ds[3].to(device)
    agg_sv = train_ds[4].to(device)
    agg_fv = train_ds[5].to(device)
    mask = train_ds[6].to(device)
    weights_sv = weights_sv.to(device)
    weights_fv = weights_fv.to(device)
    min_loss = 1e5

    for i in range(len(lengths)):
        fv_seq = fv_sequences[i, :lengths[i]]
        dec_idx = torch.where(fv_seq == 0)[0]
        weights_fv[i, dec_idx, :] = 10

    for epoch in range(500):
        dmm.train()
        sv_preds, fv_preds, kl_loss = dmm(sv_sequences, fv_sequences,
                                          obs, agg_sv, agg_fv,
                                          mask, weights_sv)

        ce_loss = torch.mean(F.binary_cross_entropy(sv_preds, sv_sequences,
                             reduction='none', weight=weights_sv) * mask) + \
                  torch.mean(F.binary_cross_entropy(fv_preds, fv_sequences,
                             reduction='none', weight=weights_fv) * mask)

        anneal_loss_w = 100
        loss = ce_loss + kl_loss * anneal_loss_w

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(dmm.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        # scheduler.step()

        if epoch % 1 == 0:
            print('Epoch {}, ce {:.4f}, kl {:.4f}'.format(epoch, ce_loss.item(), kl_loss.item()))
            test_loss = test(dmm, test_ds, device)

            if test_loss < min_loss:
                min_loss = test_loss
                torch.save(dmm, 'best_model.pth')
                print('Best model saved!')


def main():
    setup_seed(42)

    path = 'sample_data/dmchm_data.pkl'
    train_ds, test_ds = sequence_dataloader(path)

    device = 'cpu'
    dmm = DMCHM().to(device)

    optimizer = torch.optim.Adam(dmm.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=1000)
    mask = train_ds[-1]
    lengths = train_ds[2]

    weights_sv = torch.ones(mask.shape)
    weights_fv = torch.ones(mask.shape)
    for i in range(len(lengths)):
        weights_sv[i, lengths[i] - 1, :] = 10

    train(dmm, optimizer, scheduler, train_ds, test_ds, weights_sv, weights_fv, device)


if __name__ == "__main__":
    main()
