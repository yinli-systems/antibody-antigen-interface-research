# Optional multimodal source stack

These three modules are exact byte copies of source commit `35dfab14143578c75b7aa7c838cff8a5d2215231`, relocated together so their relative imports retain the original optional-feature implementation. The canonical parent modules retain their already-registered frozen source bytes. This avoids changing historical protocol hashes or dropping the optional implementation to make tests pass.

The root `multimodal_pair_aware_esm2.py` re-exports the four public classes. The optional stack and frozen core have separate tests. No historical performance is reassigned to this integrated namespace; use the original commit for exact experimental reproduction.
