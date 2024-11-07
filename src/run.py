import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime, timedelta

from lib.utils import set_random_seed, setup_logging, adjusted_time
from lib.dataloader import load_data
from lib.model_utils import model, eval_data, train2


# Plus 13 hrs
OmegaConf.register_new_resolver("adjusted_time", adjusted_time)


@hydra.main(version_base=None, config_path="config", config_name="ALACPD")
def main(cfg: DictConfig) -> None:

    # Set random seed
    set_random_seed(cfg.seed)

    # Load the data
    data, annotations, time = load_data(cfg)

    # Initialize model and training settings
    logging.info(
        "#########################################################################################"
    )
    logging.info(
        "#########                   Generate Models and Offline Training                 ########"
    )
    logging.info(
        "#########################################################################################"
    )
    cpdnet_init, Data, cpdnet, cpdnet_tensorboard = model(cfg)

    logging.info(
        "#########################################################################################"
    )
    logging.info(
        "#########                   Calculate Mean Loss on Normal Data                  ########"
    )
    logging.info(
        "#########################################################################################"
    )

    loss_normal = eval_data(
        cfg["ensemble_space"],
        cpdnet,
        Data[0].train[0],
        Data[0].train[1],
        device="cuda:7",
    )

    logging.info("Loss Normal Data = %s", loss_normal)

    # Online training loop
    logging.info(
        "#########################################################################################"
    )
    logging.info(
        "#########                             Online Training                            ########"
    )
    logging.info(
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
    data_loss = np.zeros((data.shape[0], cfg["ensemble_space"]))
    data_mean_loss = np.zeros((data.shape[0], cfg["ensemble_space"]))

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
        l_test_i = eval_data(
            cfg["ensemble_space"],
            cpdnet,
            x_in,
            y_in,
            device="cuda:7",
        )
        data_loss[idx, :] = l_test_i

        logging.info(
            f"X_test {idx} --> mean loss= {np.mean(l_test_i)} loss: {l_test_i}"
        )

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
            logging.info("#### Anomaly Detected ####")
            y_ano[m_train + i] = 1
            counter_ano += 1
            ano_indices_train.append(i)
            ano_indices_plot.append(idx)

            if counter_ano > num_ano_cpd:
                logging.info(
                    "#########################################################################################"
                )
                logging.info("######### Change-point Detected ########")
                logging.info(
                    "#########################################################################################"
                )

                # Insert change-point index and retrain model
                cpd_indices.append(idx - num_ano_cpd + int(cpdnet_init[0].window / 2))
                logging.info(
                    f"cpd point: {idx - num_ano_cpd + int(cpdnet_init[0].window / 2)}"
                )
                logging.info(f"Train on: {ano_indices_train}")
                logging.info(f"Train on (plot): {ano_indices_plot}")

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
                cpdnet = train2(
                    x,
                    y,
                    cfg["ensemble_space"],
                    cpdnet,
                    cpdnet_init,
                    tensorboard=cpdnet_tensorboard,
                    epochs=cfg["epochs_to_train_after_cpd"],
                    device="cuda:7",
                )

                loss_normal = eval_data(cfg["ensemble_space"], cpdnet, x, y)
                logging.info(f"New loss = {loss_normal}")

                ano_indices_train = []

                logging.info("Model Adapted to new data.")
            data_mean_loss[idx, :] = loss_normal

        else:
            counter_ano = 0
            ano_indices_train = []
            m_normal += 1

            # Determine if extra samples have been reached
            if m_normal > cfg["extra_samples_after_cpd"]:
                # Train with PyTorch `train2` function
                cpdnet = train2(
                    x_in,
                    y_in,
                    cfg["ensemble_space"],
                    cpdnet,
                    cpdnet_init,
                    tensorboard=cpdnet_tensorboard,
                    epochs=cfg["epochs_to_train_single_sample"],
                    device="cuda:7",
                )

                # Evaluate the ensemble using the `eval_data` function (modified for PyTorch)
                l_test_i_n = eval_data(
                    cfg["ensemble_space"], cpdnet, x_in, y_in, device="cuda:7"
                )

                # Update mean loss for normal data
                loss_normal = (loss_normal * (m_normal - 1) + l_test_i_n) / m_normal
                logging.info("New average loss of normal data = %s", loss_normal)

            else:
                # Train without updating loss in this branch
                cpdnet = train2(
                    x_in,
                    y_in,
                    cfg["ensemble_space"],
                    cpdnet,
                    cpdnet_init,
                    tensorboard=cpdnet_tensorboard,
                    epochs=cfg["epochs_to_train_single_sample"],
                    device="cuda:7",
                )

                # Evaluate the ensemble
                l_test_i_n = eval_data(
                    cfg["ensemble_space"], cpdnet, x_in, y_in, device="cuda:7"
                )

                # Update mean loss for normal data
                loss_normal = (loss_normal * (m_normal - 1) + l_test_i_n) / m_normal
                logging.info("new average loss of normal data = %s", loss_normal)

            # Store the updated loss in `data_mean_loss`
            data_mean_loss[idx, :] = loss_normal

            # Save results and analysis


import os

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("ALACPD")
    # print current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    main()
