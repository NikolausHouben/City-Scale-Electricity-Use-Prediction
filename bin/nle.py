from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    Constraint,
    Objective,
    Reals,  # type: ignore
    NonNegativeReals,  # type: ignore
)
from pyomo.opt import SolverFactory
import pandas as pd
from datetime import timedelta
import os
import sys
import re
import argparse
import json
import pickle
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import (
    infer_frequency,
    create_directory,
)
import numpy as np
import wandb

from utils.paths import ROOT_DIR, EVAL_DIR, RESULTS_DIR

from utils.model_utils import Config


def run_opt(load_forecast, bss_energy, peak, config, prices=None):
    ################################
    # MPC optimization model
    ################################
    m = ConcreteModel()
    m.T = Set(initialize=list(range(len(load_forecast))))

    # MPC Params
    # m.energy_prices = Param(m.T, initialize=prices)
    m.demand = Param(m.T, initialize=load_forecast)
    m.bss_energy_start = Param(initialize=bss_energy)
    m.rel_peak = Param(initialize=peak)

    # Config Params
    m.bss_size = Param(initialize=config.bat_size_kwh)
    m.bss_eff = Param(initialize=config.bat_efficiency)
    m.bss_max_pow = Param(initialize=config.bat_max_power)
    m.bss_end_soc_weight = Param(initialize=config.bat_end_soc_weight)
    m.peak_cost = Param(initialize=config.peak_cost)

    # variables
    m.net_load = Var(m.T, domain=Reals)
    m.peak = Var(domain=NonNegativeReals)
    m.dev_peak_plus = Var(domain=NonNegativeReals)
    m.dev_peak_minus = Var(domain=NonNegativeReals)
    m.bss_p_ch = Var(m.T, domain=Reals)
    m.bss_en = Var(m.T, domain=NonNegativeReals)

    def energy_balance(m, t):
        return m.net_load[t] == m.bss_eff * m.bss_p_ch[t] + m.demand[t]

    m.energy_balance = Constraint(m.T, rule=energy_balance)

    def operation_peak(m, t):
        return m.peak >= m.net_load[t]

    m.operation_peak = Constraint(m.T, rule=operation_peak)

    def relevant_peak(m):
        return m.peak - m.rel_peak == m.dev_peak_plus - m.dev_peak_minus

    m.relevant_peak = Constraint(rule=relevant_peak)

    def bat_soc(m, t):
        if t == 0:
            return m.bss_en[t] == m.bss_energy_start + m.bss_p_ch[t]
        else:
            return m.bss_en[t] == m.bss_en[t - 1] + m.bss_p_ch[t]

    m.bat_soc = Constraint(m.T, rule=bat_soc)

    def bat_lim_energy(m, t):
        return m.bss_en[t] <= m.bss_size

    m.bat_lim_energy = Constraint(m.T, rule=bat_lim_energy)

    def bat_lim_power_pos(m, t):
        return m.bss_p_ch[t] <= m.bss_max_pow

    m.bat_lim_power_pos = Constraint(m.T, rule=bat_lim_power_pos)

    def bat_lim_power_neg(m, t):
        return -m.bss_max_pow <= m.bss_p_ch[t]

    m.bat_lim_power_neg = Constraint(m.T, rule=bat_lim_power_neg)

    def cost(m):
        terminal_cost_weight = m.bss_end_soc_weight
        final_soc = m.bss_en[len(m.T) - 1] / m.bss_size
        terminal_cost = (final_soc - m.bss_energy_start) ** 2
        peak_costs = (m.dev_peak_plus + m.rel_peak) * m.peak_cost

        total_costs = (
            1 - terminal_cost_weight
        ) * peak_costs + terminal_cost_weight * terminal_cost
        return total_costs

    m.ObjectiveFunction = Objective(rule=cost)

    opt = SolverFactory("cplex")
    results = opt.solve(m)

    # get results
    res_df = pd.DataFrame(index=list(m.T))
    for v in [m.net_load, m.bss_en, m.bss_p_ch]:
        res_df = res_df.join(pd.Series(v.get_values(), name=v.getname()))

    res_df["load_forecast"] = list(m.demand._data.values())
    # fig = px.line(res_df, title="MPC Results")
    # fig.show()

    sp = res_df.iloc[0].to_dict()

    return sp


def run_operations(dfs_mpc, config):
    print("Running operations")
    # function to run operations of given a forecast and system data

    # initializing peak (it will store the historical peak)
    # initialize energy in the battery with the initial soc
    energy_in_the_battery = config.bat_size_kwh * config.bat_initial_soc

    operations = {}
    for t, df_mpc in enumerate(dfs_mpc[:-1]):
        load_forecast = df_mpc.iloc[:, 0].values.tolist()
        load_ground_truth = df_mpc.iloc[:, 1].values.tolist()

        if t == 0:
            peak = sum(load_forecast) / len(load_forecast)
        else:
            peak = min(min(opr_net_load, peak), sum(load_forecast) / len(load_forecast))

        sp = run_opt(
            load_forecast=load_forecast,
            bss_energy=energy_in_the_battery,
            peak=peak,
            config=config,
        )

        set_point = {}

        opr_net_load = (
            load_ground_truth[0] + sp["bss_p_ch"]
        )  # update the ground truth with the set point

        set_point.update(
            {
                "load_actual": load_ground_truth[0],
                "load_forecast": load_forecast[0],
                "opt_net_load": sp["net_load"],
                "opr_net_load": opr_net_load,
                "bss_en": sp["bss_en"],
                "bss_p_ch": sp["bss_p_ch"],
            }
        )

        # update energy in the battery
        energy_in_the_battery = sp["bss_en"]

        operations.update({t: set_point})

    df_operations = pd.DataFrame(operations).T

    df_operations.index = pd.date_range(
        dfs_mpc[0].index[1], periods=len(df_operations), freq="H"
    )

    return df_operations


def calculate_nle_stats(df_op, config):
    # ex-post cost calculation

    nle_stats = {}

    # the difference between the peak of the load and the net load for each day
    peaks_diff = (
        df_op.loc[df_op["load_actual"].groupby(df_op.index.day).idxmax().values][
            ["opr_net_load", "load_actual"]
        ]
        .diff(axis=1)
        .max()[1]
    )

    nle_stats["peak_reward"] = peaks_diff * config.peak_cost
    # checking if the bess is properly utilized
    nle_stats["bss_en_spread"] = df_op["bss_en"].max() - df_op["bss_en"].min()

    return nle_stats


def run_nle(eval_dict, scale, location, horizon, season, model):
    print(f"Running NLE for Horizon: {horizon} and Season: {season}")

    # creating directory for results
    MPC_RESULTS_DIR = os.path.join(RESULTS_DIR, "mpc_results", scale, location)
    create_directory(MPC_RESULTS_DIR)

    # loading the nle_config
    with open(os.path.join(ROOT_DIR, "nle_config.json"), "r") as fp:
        nle_config = json.load(fp)
        nle_config = Config.from_dict(nle_config, is_initial_config=False)

    wandb.config.update(nle_config.data)

    # getting predictions for the given horizon and season
    ts_list_per_model = eval_dict[horizon][season][
        0
    ]  # idx=0: ts_list (see eval_utils.py)

    # getting the ground truth
    gt_series = eval_dict[horizon][season][2]
    df_gt = gt_series.pd_dataframe()  # idx=2: groundtruth (see eval_utils.py)
    df_gt.columns = ["gt_" + df_gt.columns[0]]  # rename column to avoid confusion

    # grabbing stats for scaling the predictions and ground truth -> energy prices are a function of the ground truth
    gt_max = df_gt.max().values[0]
    gt_min = df_gt.min().values[0]

    print(f"Running MPC for {model}")

    # Ground truth operations and costs as baseline
    dfs = [
        (
            (
                (
                    ts_forecast.pd_dataframe()
                    .join(df_gt, how="left")
                    .join(
                        df_gt.rename({df_gt.columns[0]: "gt_sup"}, axis=1), how="left"
                    )  # adding gt twice so when we use .iloc[:, 1:] we get the gt, gt for the baseline
                )
                - gt_min
            )
            / (gt_max - gt_min)
        )
        for ts_forecast in ts_list_per_model[model]
    ]

    dfs_mpc_forecast = [df_mpc.iloc[:, :2] for df_mpc in dfs]

    dfs_mpc_gt = [df_mpc.iloc[:, 1:] for df_mpc in dfs]

    dfs_mpc_dict = {"gt": dfs_mpc_gt, "forecast": dfs_mpc_forecast}

    nle_stats_dict = {}
    dfs_operations = []

    for key, dfs_mpc in dfs_mpc_dict.items():
        # running operations
        df_operations = run_operations(dfs_mpc, nle_config)

        # calculating nle stats
        nle_stats = calculate_nle_stats(df_operations, nle_config)

        df_operations.columns = [f"{key}_{col}" for col in df_operations.columns]
        dfs_operations.append(df_operations)
        nle_stats_dict.update({key: nle_stats})

    df_nle_stats = pd.DataFrame(nle_stats_dict).T.reset_index()
    df_op = pd.concat(dfs_operations, axis=1)

    fig = px.line(df_op, title=f"{location} - {season} - {horizon} - {model}")
    wandb.log({"nle_operations": fig})

    wandb.log({"nle_stats": wandb.Table(dataframe=df_nle_stats)})

    nle_score = (
        nle_stats_dict["gt"]["peak_reward"] - nle_stats_dict["forecast"]["peak_reward"]
    )

    wandb.log({"nle_score": nle_score})

    return nle_score


def main():
    parser = argparse.ArgumentParser(description="Run MPC")
    parser.add_argument("--scale", type=str, help="Spatial scale", default="1_county")
    parser.add_argument("--location", type=str, help="Location", default="Los_Angeles")
    parser.add_argument("--season", type=str, help="Winter or Summer", default="Summer")
    parser.add_argument("--horizon", type=int, help="MPC horizon", default=48)
    parser.add_argument("--model", type=str, default="RandomForest")
    args = parser.parse_args()

    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="nle")
    with open(os.path.join(EVAL_DIR, f"{args.scale}/{args.location}.pkl"), "rb") as f:
        eval_dict = pickle.load(f)

    costs = run_nle(
        eval_dict, args.scale, args.location, args.horizon, args.season, args.model
    )


if __name__ == "__main__":
    main()
