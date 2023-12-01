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
    m.demand = Param(m.T, initialize=load_forecast)
    m.bss_energy_start = Param(initialize=bss_energy)
    m.rel_peak = Param(initialize=peak)

    # Config Params
    m.bss_size = Param(initialize=config.bat_size_kwh)
    m.bss_eff = Param(initialize=config.bat_efficiency)
    m.bss_max_pow = Param(initialize=config.bat_max_power)
    m.bss_end_soc_weight = Param(initialize=config.bat_end_soc_weight)

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
        bss_en_end = m.bss_en[len(m.T) - 1]
        terminal_cost = (bss_en_end - m.bss_energy_start) ** 2

        total_costs = (
            1 - m.bss_end_soc_weight
        ) * m.peak + m.bss_end_soc_weight * terminal_cost

        return total_costs

    m.ObjectiveFunction = Objective(rule=cost)

    opt = SolverFactory("cplex")
    results = opt.solve(m)

    # get results
    res_df = pd.DataFrame(index=list(m.T))
    for v in [m.net_load, m.bss_en, m.bss_p_ch]:
        res_df = res_df.join(pd.Series(v.get_values(), name=v.getname()))

    res_df["load_forecast"] = list(m.demand._data.values())

    sp = res_df.iloc[0].to_dict()

    return sp


def run_operations(dfs_mpc, config, key):
    print(f"Running operations for {key}")
    # function to run operations of given a forecast and system data

    # initialize energy in the battery with the initial soc
    energy_in_the_battery = config.bat_size_kwh * config.bat_initial_soc

    operations = {}
    for t, df_mpc in enumerate(dfs_mpc[:-1]):
        # checks:
        assert df_mpc.shape[0] == config.horizon, "Faulty horizon"

        # count the number of nan values in the forecast
        nan_count = df_mpc.isna().sum().sum()

        # get the load forecast and ground truth
        load_forecast = df_mpc.iloc[:, 0].values.tolist()
        load_ground_truth = df_mpc.iloc[:, 1].values.tolist()
        if key == "gt":
            load_forecast = load_ground_truth

        if t == 0:
            peak = sum(load_forecast) / len(load_forecast)  # peak initialisation
        else:
            peak = min(min(opr_net_load, peak), sum(load_forecast) / len(load_forecast))  # type: ignore

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
    # get the maximum peak of the ground truth per day
    peaks = df_op["load_actual"].groupby(df_op.index.day).max()
    # only consider peaks with 'high' load
    relevant_idx_mask = peaks > df_op["load_actual"].quantile(0.8)
    peak_idx = df_op["load_actual"].groupby(df_op.index.day).idxmax()[relevant_idx_mask]
    relevant_peaks = df_op.loc[peak_idx][["opr_net_load", "load_actual"]]
    peaks_diff = relevant_peaks["load_actual"] - relevant_peaks["opr_net_load"]
    nle_stats["peak_reward"] = peaks_diff.mean() * config.peak_cost

    # the number of cycles of the battery

    full_charge = np.isclose(
        df_op["bss_en"].values,
        np.ones(len(df_op)) * config.bat_size_kwh,
        atol=1e-1,
        rtol=0,
    ).sum()
    full_discharge = np.isclose(
        df_op["bss_en"].values, np.zeros(len(df_op)), atol=1e-2, rtol=0
    ).sum()

    cycles = (full_charge + full_discharge) / 2

    nle_stats["bss_cycles"] = cycles

    return nle_stats


def run_nle(eval_dict, scale, location, horizon, season, model):
    print(f"Running NLE for Horizon: {horizon} and Season: {season}")

    # creating directory for results
    MPC_RESULTS_DIR = os.path.join(RESULTS_DIR, "nle_results", scale, location)
    create_directory(MPC_RESULTS_DIR)

    # loading the nle_config
    with open(os.path.join(ROOT_DIR, "nle_config.json"), "r") as fp:
        nle_config = json.load(fp)
        nle_config = Config.from_dict(nle_config, is_initial_config=False)
        nle_config.horizon = horizon
        nle_config.season = season
        nle_config.model = model
        nle_config.scale = scale
        nle_config.location = location

    # wandb.config.update(nle_config.data)

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
    assert gt_max > gt_min, "Ground truth max is smaller than min or equal"

    print(f"Running MPC for {model}")

    # Ground truth operations and costs as baseline
    dfs = [
        (
            ((ts_forecast.pd_dataframe().join(df_gt, how="left")) - gt_min).fillna(
                method="pad"
            )
            / (gt_max - gt_min)
        )
        for ts_forecast in ts_list_per_model[model]
    ]

    dfs_mpc_dict = {"gt": dfs, "forecast": dfs}

    nle_stats_dict = {}
    dfs_operations = []

    for key, dfs_mpc in dfs_mpc_dict.items():
        # running operations
        df_operations = run_operations(dfs_mpc, nle_config, key)

        # calculating nle stats
        nle_stats = calculate_nle_stats(df_operations, nle_config)

        df_operations.columns = [f"{key}_{col}" for col in df_operations.columns]
        dfs_operations.append(df_operations)
        nle_stats_dict.update({key: nle_stats})

    df_nle_stats = pd.DataFrame(nle_stats_dict).T.reset_index()
    df_op = pd.concat(dfs_operations, axis=1)

    nle_score = (
        nle_stats_dict["gt"]["peak_reward"] - nle_stats_dict["forecast"]["peak_reward"]
    )

    return nle_score, df_op, df_nle_stats


def main():
    parser = argparse.ArgumentParser(description="Run MPC")
    parser.add_argument("--scale", type=str, help="Spatial scale", default="1_county")
    parser.add_argument("--location", type=str, help="Location", default="Los_Angeles")
    parser.add_argument("--season", type=str, help="Winter or Summer", default="Summer")
    parser.add_argument("--horizon", type=int, help="MPC horizon", default=48)
    parser.add_argument("--model", type=str, default="LinearRegressionModel")
    parser.add_argument("--wandb_mode", type=str, default="dryrun")
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = args.wandb_mode

    wandb.init(
        project="nle",
        name=f"{args.scale}_{args.location}_{args.horizon}_{args.season}_{args.model}",
    )
    with open(os.path.join(EVAL_DIR, f"{args.scale}/{args.location}.pkl"), "rb") as f:
        eval_dict = pickle.load(f)

    nle_score, df_operations, df_nle_stats = run_nle(
        eval_dict, args.scale, args.location, args.horizon, args.season, args.model
    )

    wandb.log({"nle_score": nle_score})
    fig = px.line(df_operations)
    wandb.log({"nle_operations": fig})
    wandb.log({"nle_stats": wandb.Table(dataframe=df_nle_stats)})

    df_operations.to_csv(
        os.path.join(
            RESULTS_DIR,
            "nle_results",
            args.scale,
            args.location,
            f"{args.horizon}_{args.season}_{args.model}_operations.csv",
        )
    )
    df_nle_stats.to_csv(
        os.path.join(
            RESULTS_DIR,
            "nle_results",
            args.scale,
            args.location,
            f"{args.horizon}_{args.season}_{args.model}_stats.csv",
        )
    )


if __name__ == "__main__":
    main()
