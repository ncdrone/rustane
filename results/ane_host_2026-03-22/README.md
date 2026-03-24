# Host-Side ANE Findings on macOS 26.3

This directory captures host-side Apple Neural Engine findings on this Mac.
It started with `ipsw-safe`-based extraction and was later extended with direct
Hopper/MCP validation passes.

Targets analyzed:

- `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService`
- `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANEStorageMaintainer.xpc/Contents/MacOS/ANEStorageMaintainer`

Private frameworks confirmed present on this system:

- `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework`
- `/System/Library/PrivateFrameworks/Espresso.framework`
- `/System/Library/PrivateFrameworks/NeuralNetworks.framework`

## Why This Matters for `rustane`

Yes, these findings are relevant to the current `m3-ultra` branch.

### 1. They confirm the real Apple compile stack is MIL/MLIR/Espresso -> ANECompiler

`ANECompilerService` exposes compiler classes and strings for:

- `CoreML`
- `MIL`
- `MLIR`
- `ANECIR`
- `LLIR bundle`
- `Espresso` translation

This aligns with `rustane`'s current strategy of generating MIL graphs directly,
instead of going through CoreML as a black box. The host stack is not bypassing
MIL; it is still a first-class intermediate.

### 2. They confirm the low-level compile entrypoints we care about

Imported symbols from `ANECompiler.framework`:

- `_ANECCompile`
- `_ANECCompileJIT`
- `_ANECCompileOnline`
- `_ANECCreateModelDictionary`

That is strong evidence that the compile failures `rustane` sees are happening at
the same host compiler boundary Apple uses internally.

### 3. They show Apple has an explicit model-cache layer around compilation

The compiler and storage services expose:

- `_ANEModelCacheManager`
- `_ANEInMemoryModelCacheManager`
- `_ANEStorageHelper`
- `_ANEStorageMaintainer`

and cache-related selectors/strings like:

- `cachedModelPathFor:csIdentity:`
- `cacheURLIdentifierForModel:useSourceURL:withReply:`
- `cachedSourceModelStoreNameFor:`
- `retainModelCache=%d`
- `_ANED_MODELCACHE_GC`
- `_ANED_PURGE_COMPILED_MODEL`

This suggests there may be host-side compile-cache and retention behaviors that
`rustane` is currently not taking advantage of. If `ane-bridge` compiles from
fresh temporary model paths every time, it may be forfeiting reuse opportunities
Apple's own stack expects.

### 4. They show sandbox extensions are part of the normal compiler contract

Relevant selectors/imports:

- `compileModelAt:csIdentity:sandboxExtension:...`
- `sandboxExtensionPathForModelURL:`
- `sandbox_extension_issue_file`
- `sandbox_extension_consume`
- `sandbox_extension_release`
- `Failed to enter sandbox: %s`

This is important because `rustane` relies on reverse-engineered direct ANE
execution. If later work tries to mirror more of Apple's higher-level compiler
service flow, sandbox extension handling is not optional.

### 5. They explain some shape/format vocabulary useful for experiments

Interesting constants and strings:

- `kANEFModelTypeKey`
- `kANEFModelCoreMLValue`
- `kANEFModelMILValue`
- `kANEFModelMLIRValue`
- `kANEFModelANECIRValue`
- `kANEFModelLLIRBundleValue`
- `kANEModelKeyEspressoTranslationOptions`
- `defaultMILFileName`
- `defaultANECIRFileName`
- `model.espresso.net`
- `model.llir.bundle`

That gives us host-side names for formats and options that can guide targeted
experiments when `rustane` hits compiler behavior that is not obvious from MIL
alone.

## Main Caution

These findings do **not** directly solve the current kernel-shape issues in
`rustane`. They help with:

- understanding where the real compiler boundary is,
- understanding cache/sandbox behavior,
- choosing experiments and logging targets,
- identifying the next private frameworks to inspect.

They do **not** yet reveal why a specific MIL graph shape fails or which exact
ANE compiler pass rejects it.

## Files

- `selector_class_inventory.md`
- `cache_data_flow_map.md`
- `interesting_symbols_strings.md`
- `framework_checks.md`
- `framework_checks_part2.md`
- `framework_checks_part3.md`
- `framework_checks_part4.md`
- `rustane_hypotheses_memo.md`
- `hopper_validation_pass1.md`
- `hopper_validation_pass2.md`
- `hopper_validation_pass3.md`
- `hopper_validation_pass4.md`
- `hopper_validation_pass5.md`
- `hopper_validation_pass6_mil_vs_espresso.md`
- `hopper_validation_pass7_anecompiler.md`
- `hopper_validation_pass8_anecompiler_narrow.md`
- `hopper_validation_pass9_anecprepare.md`
- `hopper_validation_pass10_createprepare_mil.md`
- `hopper_validation_pass11_mil_metadata_helpers.md`
- `hopper_validation_pass12_validate_derived_mil.md`
- `hopper_validation_pass13_validate_mil_conversion.md`
- `hopper_validation_pass14_validate_op_list.md`
- `hopper_validation_pass15_get_mil_conversion_maps.md`
- `hopper_validation_pass16_sdpa_handler.md`
- `hopper_validation_pass17_sdpa_validation_lowering.md`
- `hopper_validation_pass18_resource_status_descriptor.md`
- `hopper_validation_pass19_zin_validation_context.md`
- `hopper_validation_pass20_tensor_dimension_legalizer.md`
- `hopper_validation_pass21_try_split_by_space.md`
- `hopper_validation_pass22_tile_with_global_refinement.md`
- `hopper_validation_pass23_tile_subgraph.md`
- `hopper_validation_pass24_backprop_tiling.md`
- `hopper_validation_pass25_subgraph_identification.md`
- `hopper_validation_pass26_pressure_region_refinement.md`
- `hopper_validation_pass27_pressure_math.md`
- `hopper_validation_pass28_subgraph_construction.md`
- `hopper_validation_pass29_boundary_predicates.md`
- `hopper_validation_pass30_cluster_constraints.md`
- `hopper_validation_pass31_spatial_policy.md`
- `hopper_validation_pass32_shared_constructor_slot.md`
- `hopper_validation_pass33_legalization_triggers.md`
- `hopper_validation_pass34_matmul_path.md`
- `hopper_validation_pass35_cache_identity.md`
- `hopper_validation_pass36_espresso_cache_flow.md`
- `hopper_validation_pass37_fill_output_format_and_channel.md`
- `hopper_validation_pass38_fill_ne_unit_info.md`
- `hopper_validation_pass39_matrixmult_unit_layer.md`
- `hopper_validation_pass40_create_matmul_layer_failures.md`
- `hopper_validation_pass41_cache_manager_lookup.md`
- `hopper_validation_pass42_request_dispatch_paths.md`
- `hopper_validation_pass43_espresso_request_construction.md`
- `hopper_validation_pass44_eval_and_mapping.md`
- `hopper_validation_pass45_request_reuse.md`
