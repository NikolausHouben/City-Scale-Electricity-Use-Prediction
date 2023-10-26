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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import (
    infer_frequency,
    create_directory,
)
import numpy as np
import wandb

from utils.paths import ROOT_DIR

from utils.model_utils import Config


def run_opt(load_forecast, prices, bss_energy, peak, config):
    ################################
    # MPC optimization model
    ################################
    m = ConcreteModel()
    m.T = Set(initialize=list(range(len(load_forecast))))

    # MPC Params
    m.energy_prices = Param(m.T, initialize=prices)
    m.demand = Param(m.T, initialize=load_forecast)
    m.bss_energy_start = Param(initialize=bss_energy)
    m.monthly_peak = Param(initialize=peak)

    # Config Params
    m.bss_size = Param(initialize=config.bss_size)
    m.bss_eff = Param(initialize=config.bss_eff)
    m.bss_max_pow = Param(initialize=config.bss_size / config.bss_duration)
    m.tier_load_magnitude = Param(initialize=config.tier_load_magnitude)
    m.tier2_multiplier = Param(initialize=config.tier2_multiplier)
    m.peak_cost = Param(initialize=config.peak_cost)

    # variables
    m.net_load = Var(m.T, domain=Reals)
    m.tier1_net_load = Var(m.T, domain=Reals)
    m.tier2_net_load = Var(m.T, domain=NonNegativeReals)
    m.peak = Var(domain=NonNegativeReals)
    m.dev_peak_plus = Var(domain=NonNegativeReals)
    m.dev_peak_minus = Var(domain=NonNegativeReals)
    m.bss_p_ch = Var(m.T, domain=Reals)
    m.bss_en = Var(m.T, domain=NonNegativeReals)

    def energy_balance(m, t):
        return m.net_load[t] == m.bss_eff * m.bss_p_ch[t] + m.demand[t]

    m.energy_balance = Constraint(m.T, rule=energy_balance)

    def tier_load_definition(m, t):
        return m.net_load[t] == m.tier1_net_load[t] + m.tier2_net_load[t]

    m.tier_load_definition = Constraint(m.T, rule=tier_load_definition)

    def tier_load_limit(m, t):
        return m.tier1_net_load[t] <= m.tier_load_magnitude

    m.tier_load_limit = Constraint(m.T, rule=tier_load_limit)

    def operation_peak(m, t):
        return m.peak >= m.net_load[t]

    m.operation_peak = Constraint(m.T, rule=operation_peak)

    def relevant_peak(m):
        return m.peak - m.monthly_peak == m.dev_peak_plus - m.dev_peak_minus

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
        return (
            sum(m.tier1_net_load[t] * m.energy_prices[t] for t in m.T)
            + sum(
                m.tier2_net_load[t] * m.energy_prices[t] * m.tier2_multiplier
                for t in m.T
            )
            + (m.dev_peak_plus + m.monthly_peak) * m.peak_cost
        )

    m.ObjectiveFunction = Objective(rule=cost)

    opt = SolverFactory("cplex")
    results = opt.solve(m)

    # get results
    res_df = pd.DataFrame(index=list(m.T))
    for v in [m.net_load, m.tier1_net_load, m.tier2_net_load, m.bss_p_ch, m.bss_en]:
        res_df = res_df.join(pd.Series(v.get_values(), name=v.getname()))

    # just returns the first (next time) set-point
    sp = res_df.iloc[0].to_dict()

    return sp


def get_forecasts(df, h, fc_type, horizon):
    df = df.loc[(df.index > h) & (df.index < (h + timedelta(hours=horizon + 1)))]
    fct = df[fc_type].to_list()
    return fct


def construct_fc_types(df_fc):
    fc_types = {}
    for col in df_fc.iloc[:, :-1].columns:
        horizon = re.findall(r"\d+", col)[0]
        fc_types[col] = int(horizon)

    fc_types[df_fc.columns[-1]] = max(list(fc_types.values()))
    return fc_types


def run_operations(dfs_mpc, config):
    # function to run operations of given a forecast and system data

    # initializing monthly peak (it will store the historical peak)
    peak = config.tier_load_magnitude  # initializing monthly peak
    # initialize energy in the battery with the initial soc
    energy_in_the_battery = config.bat_size_kwh * config.initial_soc

    operations = {}
    for df_mpc in dfs_mpc:  # TODO change to hours_of_simulation
        load_forecast = df_mpc.iloc[:, 0].values.tolist()
        ground_truth = df_mpc.iloc[:, 1].values.tolist()
        prices = df_mpc.iloc[:, 2].values.tolist()
        # gets a set_point for the next interval based on the MPC optimization
        # the horizon of the optimization is given by the length of the forecast
        set_point = run_opt(
            load_forecast=load_forecast,
            prices=prices,
            bss_energy=energy_in_the_battery,
            peak=peak,
            config=config,
        )

        # implement set point in time, calculate net and tier load, this
        set_point_time = t + timedelta(hours=1)
        net_load = fc.loc[set_point_time]["Ground Truth"] + set_point["bss_p_ch"]
        tier1_load = (
            net_load
            if net_load <= config.tier_load_magnitude
            else config.tier_load_magnitude
        )
        tier2_load = (
            0
            if net_load <= config.tier_load_magnitude
            else net_load - config.tier_load_magnitude
        )

        # implement peak condition
        if peak < net_load:
            peak = net_load
        set_point.update(
            {
                "opr_net_load": net_load,
                "opr_tier1_load": tier1_load,
                "opr_tier2_load": tier2_load,
            }
        )

        # update energy in the battery
        energy_in_the_battery = set_point["bss_en"]

        operations.update({set_point_time: set_point})

    return pd.DataFrame(operations).T


def scale_by_gt(df):
    """Scale the predictions and ground truth by the max and min of the ground truth"""

    gt_max = df["Ground Truth"].max()
    gt_min = df["Ground Truth"].min()

    df_scaled = df.copy()
    df_scaled["Ground Truth"] = (df_scaled["Ground Truth"] - gt_min) / (gt_max - gt_min)

    for col in df_scaled.columns:
        if col != "Ground Truth":
            df_scaled[col] = (df_scaled[col] - gt_min) / (gt_max - gt_min)

    return df_scaled, gt_max, gt_min


def generate_ep_profile(df, hour_shift=3, mu=0.0, sigma=0.3):
    """Generate electricity price profiles based on the ground truth of the load"""

    timesteps_per_hour = int(infer_frequency(df) // 60)
    shift_in_timesteps = hour_shift * timesteps_per_hour
    # step 1: shift the ground truth by n hours
    series = df.iloc[:, 0]
    ep1 = series.shift(shift_in_timesteps)
    # step 2: add a random noise to it
    noise = np.random.normal(mu, sigma, len(ep1))
    ep2 = ep1 + noise
    # step 3: smooth it
    ep3 = ep2.ewm(span=timesteps_per_hour * 6).mean().fillna(method="bfill")
    ep4 = ep3.to_frame("ep")
    return ep4


def run_nle(eval_dict, config):
    ##############################################
    # input parameters
    #############################################

    ts_list_per_model = eval_dict[config.horizon][config.season][
        0
    ]  # idx=0: ts_list (see eval_utils.py)
    models = (
        config.evaluate_models
        if config.evaluate_models
        else list(ts_list_per_model.keys())
    )

    gt = eval_dict[config.horizon][config.season][
        2
    ].pd_dataframe()  # idx=2: groundtruth (see eval_utils.py)
    gt.columns = [gt.columns[0] + "_Ground_Truth"]
    gt_max = gt.max().values[0]  # for scaling the predictions
    gt_min = gt.min().values[0]

    ep = generate_ep_profile(
        gt,
        config.energy_price_shift_in_hours,
        config.energy_price_noise_mu,
        config.energy_price_noise_sigma,
    )

    # simulation (mpc) starts here
    for model in models:
        dfs_mpc = [
            ((ts_forecast.pd_dataframe().join(gt, how="left").join(ep)) - gt_min)
            / (gt_max - gt_min)
            for ts_forecast in ts_list_per_model[model]
        ]
        df_operations = run_operations(dfs_mpc, config)

    # ex-post cost calculation
    cost_results = {}
    results = results.join(df_scaled[["Ground Truth"]]).join(ep)
    for fc_type in fc_types.keys():
        tier1 = (results[f"opr_tier1_load_{fc_type}"] * results["ep"]).sum()

        tier2 = (
            results[f"opr_tier2_load_{fc_type}"] * results["ep"] * tier2_cost_multiplier
        ).sum()

        peak = results[f"opr_net_load_{fc_type}"].max() * cost_of_peak

        cost_results.update(
            {
                fc_type: {
                    "horizon_in_hours": fc_types[fc_type],
                    "tier1": tier1,
                    "tier2": tier2,
                    "peak": peak,
                    "total_cost": tier1 + tier2 + peak,
                }
            }
        )

    cost_results = pd.DataFrame(cost_results).T

    return cost_results, results


import pickle


def main():
    parser = argparse.ArgumentParser(description="Run MPC")
    parser.add_argument(
        "--spatial_scale", type=str, help="Spatial scale", default="1_county"
    )
    parser.add_argument("--location", type=str, help="Location", default="Los_Angeles")
    parser.add_argument("--season", type=str, help="Winter or Summer", default="Summer")
    parser.add_argument("--horizon", type=int, help="MPC horizon", default=24)
    parser.add_argument("--models", type=list, default=None)
    args = parser.parse_args()
    MPC_RESULTS_DIR = os.path.join(
        ROOT_DIR, "data", "results", "mpc_results", args.spatial_scale, args.location
    )
    create_directory(MPC_RESULTS_DIR)

    with open(os.path.join(ROOT_DIR, "mpc_config.json"), "r") as f:
        mpc_config = json.load(f)

    with open(f"data/evaluations/{args.spatial_scale}/{args.location}.pkl", "rb") as f:
        eval_dict = pickle.load(f)

    # grabbing config, and adding arguments to it for later use
    config = Config().from_dict(mpc_config)
    config.location = args.location
    config.spatial_scale = args.spatial_scale
    config.season = args.season
    config.horizon = args.horizon
    config.evaluate_models = args.models

    cost_results, results = run_nle(eval_dict, config)

    cost_results.to_csv(os.path.join(MPC_RESULTS_DIR, "cost_results.csv"))
    results.to_csv(os.path.join(MPC_RESULTS_DIR, "operational_results.csv"))


if __name__ == "__main__":
    main()
