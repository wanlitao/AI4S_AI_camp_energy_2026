from pipeline_common import StageConfig, run_pipeline


if __name__ == "__main__":
    run_pipeline(
        StageConfig(
            stage_name="Step1: 仅使用外生预测特征的 CatBoost 直推基线",
            use_price_history=False,
            use_actual_reconstruction=False,
            use_spike_model=False,
            use_lgb_residual=False,
            use_xgb_residual=False,
            use_weighted_ensemble=False,
        ),
        __file__,
    )
