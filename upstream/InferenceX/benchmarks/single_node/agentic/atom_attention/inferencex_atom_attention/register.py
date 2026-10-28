"""Register custom backends through vLLM's supported extension interfaces."""


def register():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "inferencex_atom_attention.backend.AtomFlyDSLMLABackend",
    )
    register_mla_prefill_backend(
        MLAPrefillBackendEnum.CUSTOM,
        "inferencex_atom_attention.backend.AtomFlyDSLPrefillBackend",
    )
