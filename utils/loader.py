import numpy as np
import pickle


def sequence_dataloader(path="dmchm_data.pkl"):
    data = open(path, 'rb')
    data = pickle.load(data)

    train_idx = np.random.choice(range(len(data['label'])), int(0.8 * len(data['label'])), replace=False)
    test_idx = np.setdiff1d(np.array([i for i in range(len(data['label']))]), train_idx)

    # train set
    sv_sequences = data['label'][train_idx, :, 0].unsqueeze(-1).float()
    fv_sequences = data['label'][train_idx, :, 1].unsqueeze(-1).float()
    lengths = data['data_len'][train_idx].long()

    obs = data['obs'][train_idx].float()
    agg_sv = data['ttc_sv'][train_idx].float()
    agg_fv = data['ttc_fv'][train_idx].float()
    mask = data['mask'][train_idx].unsqueeze(-1)

    # test set
    sv_sequences_test = data['label'][test_idx, :, 0].unsqueeze(-1).float()
    fv_sequences_test = data['label'][test_idx, :, 1].unsqueeze(-1).float()
    lengths_test = data['data_len'][test_idx].long()
    obs_test = data['obs'][test_idx].float()
    agg_sv_test = data['ttc_sv'][test_idx].float()
    agg_fv_test = data['ttc_fv'][test_idx].float()
    loc = data['loc'][test_idx].float()
    mask_test = data['mask'][test_idx].unsqueeze(-1)

    return (sv_sequences, fv_sequences, lengths, obs, agg_sv, agg_fv, mask), \
           (sv_sequences_test, fv_sequences_test, lengths_test, obs_test, agg_sv_test, agg_fv_test, loc, mask_test)

