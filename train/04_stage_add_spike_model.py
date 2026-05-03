from pipeline_common import StageConfig, run_pipeline


if __name__ == "__main__":
    run_pipeline(
        StageConfig(
            stage_name="Step4: 加入 spike 分类与 spike 回归软融合",
            use_price_history=True,
            use_actual_reconstruction=True,
            use_spike_model=True,
            use_lgb_residual=False,
            use_xgb_residual=False,
            use_weighted_ensemble=False,
        ),
        __file__,
    )
