from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    Constraint,
    Objective,
    Reals,
    NonNegativeReals,
)
from pyomo.opt import SolverFactory
import pandas as pd
from datetime import timedelta


def get_forecasts(df, h, fc_type, horizon):
    df = df.loc[(df.index > h) & (df.index < (h + timedelta(hours=horizon + 1)))]
    fct = df[fc_type].to_list()
    return fct


def run_model(
    load_forecast,
    prices,
    bss_energy,
    bss_duration,
    bss_eff,
    bss_size,
    monthly_peak,
    tier_load_magnitude,
    tier2_multiplier,
    peak_cost,
):
    ################################
    # MPC optimization model
    ################################
    m = ConcreteModel()
    m.T = Set(initialize=list(range(len(load_forecast))))

    # Params
    m.energy_prices = Param(m.T, initialize=prices)
    m.demand = Param(m.T, initialize=load_forecast)
    m.bss_size = Param(initialize=bss_size)
    m.bss_eff = Param(initialize=bss_eff)
    m.bss_energy_start = Param(initialize=bss_energy)
    m.bss_max_pow = Param(initialize=bss_size / bss_duration)
    m.monthly_peak = Param(initialize=monthly_peak)
    m.tier_load_magnitude = Param(initialize=tier_load_magnitude)
    m.tier2_multiplier = Param(initialize=tier2_multiplier)
    m.peak_cost = Param(initialize=peak_cost)

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


def run_operations(
    hours_of_simulation,
    fc,
    fc_type,
    horizon,
    bat_size_kwh,
    bat_duration,
    bss_eff,
    initial_soc,
    tier_load_magnitude,
    tier2_multiplier,
    peak_cost,
):
    # function to run operations of given a forecast and system data

    # initializing monthly peak (it will store the historical peak)
    peak = tier_load_magnitude  # initializing monthly peak

    # initialize energy in the battery with the initial soc
    energy_in_the_battery = bat_size_kwh * initial_soc

    operations = {}
    for t in fc.index[:hours_of_simulation]:
        # get load forecast as a list
        load = get_forecasts(df=fc, h=t, fc_type=fc_type, horizon=horizon)

        # get the energy prices as a list
        ep = get_forecasts(df=fc, h=t, fc_type="energy_prices_dol_kwh", horizon=horizon)

        # gets a set_point for the next interval based on the MPC optimization
        # the horizon of the optimization is given by the length of the forecast
        set_point = run_model(
            load_forecast=load,
            prices=ep,
            bss_energy=energy_in_the_battery,
            bss_size=bat_size_kwh,
            bss_duration=bat_duration,
            monthly_peak=peak,
            tier_load_magnitude=tier_load_magnitude,
            tier2_multiplier=tier2_multiplier,
            peak_cost=peak_cost,
            bss_eff=bss_eff,
        )

        # implement set point in time, calculate net and tier load
        set_point_time = t + timedelta(hours=1)
        net_load = fc.loc[set_point_time]["Actual load (kW)"] + set_point["bss_p_ch"]
        tier1_load = (
            net_load if net_load <= tier_load_magnitude else tier_load_magnitude
        )
        tier2_load = (
            0 if net_load <= tier_load_magnitude else net_load - tier_load_magnitude
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


def run_all():
    ##############################################
    # input parameters
    #############################################

    # parameters of the optimization
    initial_soc = 0.5  # initial state of charge of the battery (no unit)
    bat_size_kwh = 8  # size of the battery in kWh
    bat_duration = 2  # battery duration (Max_kW = bat_size/duration)
    bat_efficiency = 0.95  # charging and discharging efficiency of the battery
    tier2_cost_multiplier = 1.5  # cost multiplication of the tier 2 load
    cost_of_peak = 5.0  # cost of the monthly peak

    hours_of_simulation = 600

    # type of forecasts to evaluation and corresponding horizon in hours
    fc_types = {
        "XGBModel_4 Hours Ahead (kW)": 4,
        "XGBModel_8 Hours Ahead (kW)": 8,
        "NBEATSModel_4 Hours Ahead (kW)": 4,
        "NBEATSModel_8 Hours Ahead (kW)": 8,
        "Actual load (kW)_4": 4,
        "Actual load (kW)_8": 8,
    }

    # read raw forecasts and calculate the average load
    fc = pd.read_csv(
        "/Users/nikolaushouben/Desktop/WattCast/mpc/forecasts.csv", index_col=0
    )
    fc.index = pd.to_datetime(fc.index)

    # tier load magnitude is equal to the average load (any other value works)
    tier_load_magnitude = fc["Actual load (kW)"].mean()

    ##############################################
    # simulation starts here
    ##############################################

    results = pd.DataFrame()
    for fc_type in fc_types.keys():
        print(f"running {fc_type}")

        fc_result = run_operations(
            hours_of_simulation=hours_of_simulation,
            fc=fc,  # forecast table
            fc_type=fc_type,  # label of forecast type
            horizon=fc_types[fc_type],
            bat_size_kwh=bat_size_kwh,
            bat_duration=bat_duration,
            initial_soc=initial_soc,
            bss_eff=bat_efficiency,
            tier_load_magnitude=tier_load_magnitude,
            tier2_multiplier=tier2_cost_multiplier,
            peak_cost=cost_of_peak,
        )

        # add the forecast label to the forecast operation results
        for k in fc_result:
            fc_result[k + f"_{fc_type}"] = fc_result[k]
            fc_result = fc_result.drop(k, axis=1)

        # build the operation results table
        if results.shape[0] == 0:
            results = pd.concat([results, fc_result])
        else:
            results = results.join(fc_result)

    # calculate operation costs of each forecast
    cost_results = {}
    results = results.join(fc[["Actual load (kW)", "energy_prices_dol_kwh"]])
    for fc_type in fc_types.keys():
        tier1 = (
            results[f"opr_tier1_load_{fc_type}"] * results["energy_prices_dol_kwh"]
        ).sum()

        tier2 = (
            results[f"opr_tier2_load_{fc_type}"]
            * results["energy_prices_dol_kwh"]
            * tier2_cost_multiplier
        ).sum()

        peak = results[f"opr_net_load_{fc_type}"].max() * cost_of_peak

        cost_results.update(
            {
                fc_type: {
                    "horizon": fc_types[fc_type],
                    "tier1": tier1,
                    "tier2": tier2,
                    "peak": peak,
                    "total": tier1 + tier2 + peak,
                }
            }
        )
    cost_results = pd.DataFrame(cost_results).T

    # write results
    cost_results.to_csv("cost_results.csv")
    results.to_csv("operation_results.csv")


if __name__ == "__main__":
    run_all()
