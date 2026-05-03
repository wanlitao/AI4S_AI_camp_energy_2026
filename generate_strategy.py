import os
import sys

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(BASE_DIR, path_value))


def generate_strategy(price_csv, save_path):
    """
    根据预测的实时价格确定充放电策略

    策略逻辑：
    1. 寻找最优的充电开始时间 tc 和放电开始时间 td
    2. 充电持续 8 个时间点（2 小时），放电持续 8 个时间点（2 小时）
    3. 充电开始时间：0 <= tc <= 80
    4. 放电开始时间：td >= tc + 8 且 td <= 88
    5. 目标：最大化收益 = sum(放电时段价格) * 1000 - sum(充电时段价格) * 1000
    """
    df = pd.read_csv(price_csv)
    df["times"] = pd.to_datetime(df["times"])

    df["date"] = df["times"].dt.date

    results = []
    total_profit = 0

    for date, group in df.groupby("date"):
        prices = group["A"].values
        times = group["times"].values

        n = len(prices)
        if n != 96:
            print(f"警告: {date} 的数据点数量为 {n}，预期为 96")
            continue

        best_profit = 0
        best_tc = -1
        best_td = -1

        for tc in range(0, 81):  # 0 <= tc <= 80
            charge_prices = prices[tc : tc + 8]
            charge_cost = np.sum(charge_prices) * 1000

            for td in range(tc + 8, 89):  # td >= tc + 8 且 td <= 88
                discharge_prices = prices[td : td + 8]
                discharge_revenue = np.sum(discharge_prices) * 1000

                profit = discharge_revenue - charge_cost

                if profit > best_profit:
                    best_profit = profit
                    best_tc = tc
                    best_td = td

        power = np.zeros(96)
        if best_tc >= 0 and best_td >= 0:
            power[best_tc : best_tc + 8] = -1000
            power[best_td : best_td + 8] = 1000
            total_profit += best_profit
            print(
                f"日期: {date}, 充电开始 {best_tc:2d}, 放电开始 {best_td:2d}, 预期收益: {best_profit:10.2f}"
            )
        else:
            print(f"日期: {date}, 无交易（收益非正）")

        for t, p, pr in zip(times, power, prices):
            results.append({"times": t, "实时价格": pr, "power": p})

    df_result = pd.DataFrame(results)
    df_result.to_csv(save_path, index=False)

    n_days = len(df.groupby("date"))
    avg_profit = total_profit / n_days if n_days > 0 else 0

    print(f"\n充放电策略已保存: {save_path}")
    print(f"总天数: {n_days}")
    print(f"总收益: {total_profit:.2f}")
    print(f"平均日收益: {avg_profit:.2f}")

    return df_result


def main():
    if len(sys.argv) < 2:
        print("用法: python generate_strategy.py <预测电价文件路径> [充放电策略保存路径]")
        sys.exit(1)

    price_csv = resolve_path(sys.argv[1])
    save_path_arg = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "output.csv")
    save_path = resolve_path(save_path_arg)

    if not os.path.exists(price_csv):
        print(f"错误: 预测电价文件不存在: {price_csv}")
        sys.exit(1)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    generate_strategy(price_csv, save_path)
    print(f"提交文件: {save_path}")


if __name__ == "__main__":
    main()
