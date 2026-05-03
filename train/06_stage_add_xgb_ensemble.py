from pipeline_common import StageConfig, run_pipeline


if __name__ == "__main__":
    run_pipeline(
        StageConfig(
            stage_name="Step6: 继续叠加 XGBoost 残差修正与非负加权集成",
            use_price_history=True,
            use_actual_reconstruction=True,
            use_spike_model=True,
            use_lgb_residual=True,
            use_xgb_residual=True,
            use_weighted_ensemble=True,
        ),
        __file__,
    )
