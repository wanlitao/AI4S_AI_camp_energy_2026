from pipeline_common import StageConfig, run_pipeline


if __name__ == "__main__":
    run_pipeline(
        StageConfig(
            stage_name="Step3: 加入实际值重建与偏差代理特征",
            use_price_history=True,
            use_actual_reconstruction=True,
            use_spike_model=False,
            use_lgb_residual=False,
            use_xgb_residual=False,
            use_weighted_ensemble=False,
        ),
        __file__,
    )
