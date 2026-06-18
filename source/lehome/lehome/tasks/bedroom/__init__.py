import gymnasium as gym

gym.register(
    id="LeHome-BiSO101-Direct-Garment-v2",
    entry_point=f"{__name__}.garment_bi_v2:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg_v2:GarmentEnvCfg",
    },
)
