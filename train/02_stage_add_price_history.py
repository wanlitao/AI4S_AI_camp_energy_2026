from pipeline_common import StageConfig, run_pipeline


if __name__ == "__main__":
    run_pipeline(
        StageConfig(
            stage_name="Step2: 在外生基线上叠加安全价格历史特征",
            use_price_history=True,
            use_actual_reconstruction=False,
            use_spike_model=False,
            use_lgb_residual=False,
            use_xgb_residual=False,
            use_weighted_ensemble=False,
        ),
        __file__,
    )
