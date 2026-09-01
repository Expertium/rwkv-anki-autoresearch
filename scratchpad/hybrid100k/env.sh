# The champion's EXACT env (iter 53). A param map taken without these is of a
# different model: STRIP_CMIX alone removes 9 of 13 channel mixers.
export OMP_NUM_THREADS=1
export RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
export RWKV_INTERLEAVE=1
export RWKV_GRU_HEAD=3
export RWKV_PAVA_LAMBDA=0.2
export RWKV_NO_AHEAD_RESIDUAL=1
export RWKV_STRIP_L0_VLORA=1
export RWKV_ZERO_FEATURES=22
export RWKV_STATE_CLAMP_TAU=300
export RWKV_STATE_CLAMP_WINDOW=32768
export RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
