import os
import sys
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

# Your custom imports
# from plt_utils import plot_results
# from utils import *
# from model_util import model, eval_data, train2
# import pickle
# from lstmae_ensemble import ModelCompile
# from metrics import f_measure, covering
# from cpdnet_datautil import DataUtil

from lib.utils import set_random_seed
from lib.dataloader import load_data
from lib.model_utils import model


@hydra.main(version_base=None, config_path="config", config_name="ALACPD")
def main(cfg: DictConfig) -> None:

    # Set random seed
    set_random_seed(cfg.seed)

    # Load the data
    data, annotations, time = load_data(cfg)

    # Initialize model and training settings
    print(
        "#########################################################################################"
    )
    print(
        "#########                   Generate Models and Offline Training                 ########"
    )
    print(
        "#########################################################################################"
    )
    cpdnet_init, Data, cpdnet, cpdnet_tensorboard, graph, sess = model(cfg, cfg, data)

    print(
        "#########################################################################################"
    )
    print(
        "#########                   Calculate Mean Loss on Normal Data                  ########"
    )
    print(
        "#########################################################################################"
    )
    loss_normal = eval_data(
        params["ensemble_space"],
        cpdnet,
        graph,
        sess,
        Data[0].train[0],
        Data[0].train[1],
    )
    print("Loss Normal Data = ", loss_normal)

    # Online training loop
    print(
        "#########################################################################################"
    )
    print(
        "#########                             Online Training                            ########"
    )
    print(
        "#########################################################################################"
    )

    m_train = Data[0].train[0].shape[0]
    m_test = Data[0].test[0].shape[0]
    m_normal = m_train

    # Initialize result lists
    ano_indices_train = []
    ano_indices_plot = []
    cpd_indices = []
    y_ano = (m_train + m_test) * [0]
    data_loss = np.zeros((data.shape[0], params["ensemble_space"]))
    data_mean_loss = np.zeros((data.shape[0], params["ensemble_space"]))

    for i in range(cpdnet_init[0].window * 2 + (m_normal - 1)):
        data_loss[i, :] = loss_normal
        data_mean_loss[i, :] = loss_normal

    counter_ano = 0
    num_ano_cpd = cfg.num_ano_cpd
    th1 = cfg.threshold_high
    th2 = cfg.threshold_low

    # Iterate over test data
    for i in range(m_test):
        idx = cpdnet_init[0].window * 2 + (m_train + i - 1)
        x_in = np.expand_dims(Data[0].test[0][i], axis=0)
        y_in = np.expand_dims(Data[0].test[1][i], axis=0)

        # Evaluate loss
        l_test_i = eval_data(params["ensemble_space"], cpdnet, graph, sess, x_in, y_in)
        data_loss[idx, :] = l_test_i

        print(f"X_test {idx} --> mean loss= {np.mean(l_test_i)} loss: {l_test_i}")

        # Check for anomaly
        if m_normal > cfg.num_change_threshold:
            th = th2
        else:
            th = th1

        # Anomaly detection
        if (
            np.sum(
                [l_test_i[k] >= th * loss_normal[k] for k in range(cfg.ensemble_space)]
            )
            > (cfg.ensemble_space / 2)
            and i > 8
            and m_normal > 4
        ):
            print("#### Anomaly Detected ####", flush=True)
            y_ano[m_train + i] = 1
            counter_ano += 1
            ano_indices_train.append(i)
            ano_indices_plot.append(idx)

            if counter_ano > num_ano_cpd:
                print(
                    "#########################################################################################"
                )
                print("######### Change-point Detected ########")
                print(
                    "#########################################################################################",
                    flush=True,
                )

                # Insert change-point index and retrain model
                cpd_indices.append(idx - num_ano_cpd + int(cpdnet_init[0].window / 2))
                print(
                    f"cpd point: {idx - num_ano_cpd + int(cpdnet_init[0].window / 2)}"
                )
                print(f"Train on: {ano_indices_train}")
                print(f"Train on (plot): {ano_indices_plot}")

                for i in range(num_ano_cpd + 1):
                    del ano_indices_plot[-(num_ano_cpd - i)]

                counter_ano = 0

                # Train model with new samples
                x = Data[0].test[0][
                    ano_indices_train[-1] : ano_indices_train[-1]
                    + cfg.extra_samples_after_cpd
                ]
                y = Data[0].test[1][
                    ano_indices_train[-1] : ano_indices_train[-1]
                    + cfg.extra_samples_after_cpd
                ]
                m_normal = 0
                cpdnet, graph, sess = train2(
                    x,
                    y,
                    params["ensemble_space"],
                    cpdnet,
                    graph,
                    sess,
                    cpdnet_init,
                    cpdnet_tensorboard,
                    epochs=cfg.epochs_to_train_after_cpd,
                )
                loss_normal = eval_data(
                    params["ensemble_space"], cpdnet, graph, sess, x, y
                )
                print(f"New loss = {loss_normal}", flush=True)

                ano_indices_train = []
                plot_results(
                    data,
                    time,
                    data_loss,
                    data_mean_loss,
                    ano_indices_plot,
                    cpd_indices,
                    save_dir=cpdnet_init[0].logfilename,
                    name=cfg.dataset_name,
                    current_idx=idx,
                )
                print("Model Adapted to new data.")
        data_mean_loss[idx, :] = loss_normal

    # Save results and analysis
    np.savetxt(
        cfg.file_name + "cpd_indices.out", np.asarray(cpd_indices), delimiter=","
    )
    print("Final predicted change-points:", cpd_indices)

    if cfg.dataset_name in ["occupancy", "apple", "bee_waggle_6", "run_log"]:
        with open(os.path.join(cfg.file_name, "results.txt"), "w") as file1:
            print("\nReal:", annotations)
            file1.write("\nPredicted: " + str(cpd_indices))
            file1.write("\nReal: " + str(annotations))
            print("\nCovering: ", covering(annotations, cpd_indices, data.shape[0]))
            print("\nF-Measure: ", f_measure(annotations, cpd_indices))
            file1.write(
                "\nCovering: " + str(covering(annotations, cpd_indices, data.shape[0]))
            )
            file1.write("\nF-Measure: " + str(f_measure(annotations, cpd_indices)))

    print("Algorithm Finished")


if __name__ == "__main__":
    main()
