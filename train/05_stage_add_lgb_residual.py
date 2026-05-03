from pipeline_common import StageConfig, run_pipeline


if __name__ == "__main__":
    run_pipeline(
        StageConfig(
            stage_name="Step5: 在 spike 模型基础上叠加 LightGBM 残差修正",
            use_price_history=True,
            use_actual_reconstruction=True,
            use_spike_model=True,
            use_lgb_residual=True,
            use_xgb_residual=False,
            use_weighted_ensemble=False,
        ),
        __file__,
    )
