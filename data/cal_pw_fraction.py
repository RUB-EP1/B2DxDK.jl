import extra_amp
from tf_pwa.config_loader import ConfigLoader
import numpy as np
import itertools
import csv

def create_partial_wave(config):
    partial_wave = []
    print("index", "chain", "ls")
    for idx, chain in enumerate(config.get_decay()):
        ls_list = []
        all_ls = []
        for decay in chain:
            n_ls = len(decay.get_ls_list())
            name = str(decay.g_ls)
            ls_params = [f"{name}_{i}" for i in range(n_ls)]
            ls_list.append(ls_params)
            all_ls += ls_params
        order_ls = 0
        for order_ls, combination in enumerate(itertools.product(*ls_list)):
            mask_params = {}
            for i in all_ls:
                if i not in combination:
                    mask_params[i+"r"] = 0.0
                    mask_params[i+"i"] = 0.0
            print(len(partial_wave), idx, order_ls)
            partial_wave.append((idx, order_ls, mask_params))
    return partial_wave

def cal_frac(config, phsp_noeff, partial_wave):
    amp = config.get_amplitude()

    total_int = config.batch_sum_var(amp, phsp_noeff)

    pw_int = []
    for idx, _, mask_params in partial_wave:
        with amp.temp_used_res([idx]):
            with amp.mask_params(mask_params):
                tmp = config.batch_sum_var(amp, phsp_noeff)
        pw_int.append(tmp)

    pw_inf_int = {}
    for order, (idx, _, mask_params) in enumerate(partial_wave):
        for order2, (idx2, _, mask_params2) in enumerate(partial_wave):
            if order <= order2:
                continue
            if idx == idx2:
                used_res = [idx]
                new_mask_params = {k: v for k,v in mask_params.items() if k in mask_params2}
            else:
                used_res = [idx, idx2]
                new_mask_params = {k: v for k,v in mask_params.items()}
                new_mask_params.update(mask_params2)
            with amp.temp_used_res(used_res):
                with amp.mask_params(new_mask_params):
                    tmp = config.batch_sum_var(amp, phsp_noeff)
            pw_inf_int[(order, order2)] = tmp

    def get_name(idx):
        chain_idx, ls_idx, _ = partial_wave[idx]
        res = config.get_decay()[chain_idx][1].core
        return f"{res}/{ls_idx}"

    fit_frac = {}
    fit_frac_error = {}
    for i, (idx_i, ls_i, _) in enumerate(partial_wave):
        for j, (idx_j, ls_j, _) in enumerate(partial_wave):
            if i<j:
                continue
            if i==j:
                with config.params_trans() as pt:
                    tmp = pw_int[i]()/total_int()
                key = get_name(i)
            if i!=j:
                with config.params_trans() as pt:
                    tmp = (pw_inf_int[(i,j)]() - pw_int[i]() - pw_int[j]())/total_int()
                key = (get_name(i), get_name(j))
            fit_frac[key] = float(tmp)
            fit_frac_error[key] = float(pt.get_error(tmp))
    return fit_frac, fit_frac_error

def save_frac_csv(file_name, fit_frac):
    from tf_pwa.utils import tuple_table
    table = tuple_table(fit_frac)
    with open(file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(table)

def main():
    config = ConfigLoader("config_a.yml")
    config.set_params("final_params.json")
    config.inv_he = np.load("error_matrix.npy")
    partial_wave = create_partial_wave(config)
    phsp_noeff = config.get_phsp_noeff()

    fit_frac, fit_frac_e = cal_frac(config, phsp_noeff, partial_wave)
    save_frac_csv("fit_frac1_pw.csv", fit_frac)
    save_frac_csv("fit_frac1_pw_err.csv", fit_frac_e)
    # phsp_noeff["c"] = np.ones_like(phsp_noeff["c"])
    # fit_frac, fit_frac_e = cal_frac(config, phsp_noeff, partial_wave)
    # save_frac_csv("fit_frac1_pw.csv", fit_frac)
    # save_frac_csv("fit_frac1_pw_err.csv", fit_frac_e)


if __name__ == "__main__":
    main()
