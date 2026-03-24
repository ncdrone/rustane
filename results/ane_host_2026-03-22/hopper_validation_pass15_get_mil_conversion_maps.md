# Hopper Validation Pass 15: `GetMILConversionMaps`

This pass targets:

- `GetMILConversionMaps()`

The goal is to determine whether this is the compiled-in registry that
`ValidateOpList(...)` uses to map MIL op names to bespoke validation/lowering
handlers.

## 1. High-level result

Yes. `GetMILConversionMaps()` looks exactly like the hard-coded operation
registry for the converted-MIL validator.

It is very large:

- length: `15112` bytes
- basic blocks: `7`

but structurally simple:

- it constructs a long sequence of op-name strings
- builds `std::map<string, std::function<...>>` objects from them
- returns the resulting registry to `ValidateOpList(...)`

So this is not dynamic discovery. It is a compiled-in table of operation-name
handlers.

## 2. Caller relationship

Direct caller:

- `ValidateOpList(...)`

This confirms the earlier inference:

- `ValidateOpList(...)` uses this map as its central op-name -> handler lookup
  table

So when `ValidateOpList(...)` retrieves an operation identifier, this registry
is the next likely place it goes to decide how that op should be validated or
materialized.

## 3. What the registry contains

The decompiled string construction gives a very clear view of the operation
families Apple has explicit handlers for.

Representative entries include:

### 3.1 Buffer / I/O / boundary ops

- `tensor_buffer_to_tensor`
- `tensor_to_tensor_buffer`
- `circular_buffer_to_tensor`
- `tensor_to_circular_buffer`
- `pixel_buffer_to_tensor`
- `tensor_to_pixel_buffer`

### 3.2 Activation / elementwise ops

- `clamped_relu`
- `elu`
- `erf`
- `gelu`
- `leaky_relu`
- `linear_activation`
- `prelu`
- `relu`
- `relu6`
- `scaled_tanh`
- `sigmoid`
- `sigmoid_hard`
- `silu`
- `softmax`
- `softplus`
- `softplus_parametric`
- `softsign`
- `thresholded_relu`
- `abs`
- `atan`
- `ceil`
- `cos`
- `clip`
- `exp`
- `exp2`
- `floor`
- `inverse`
- `log`
- `logical_not`
- `round`
- `rsqrt`
- `sign`
- `sin`
- `sqrt`
- `square`
- `tanh`
- `threshold`

### 3.3 Arithmetic / comparison ops

- `add`
- `equal`
- `floor_div`
- `greater`
- `greater_equal`
- `less`
- `less_equal`
- `maximum`
- `minimum`
- `mul`
- `not_equal`
- `pow`
- `real_div`
- `sub`
- `mod`
- `logical_and`
- `logical_or`
- `logical_xor`

### 3.4 Convolution / matmul / normalization / pooling

- `conv`
- `conv_transpose`
- `linear`
- `matmul`
- `einsum`
- `batch_norm`
- `instance_norm`
- `l2_norm`
- `local_response_norm`
- `layer_norm`
- `avg_pool`
- `max_pool`
- `l2_pool`

### 3.5 Gather / scatter / resize / geometry / indexing

- `gather`
- `gather_along_axis`
- `gather_nd`
- `crop`
- `crop_resize`
- `resize`
- `resize_bilinear`
- `resize_nearest_neighbor`
- `upsample_nearest_neighbor`
- `upsample_bilinear`
- `affine`
- `resample`
- `scatter`
- `scatter_along_axis`
- `scatter_nd`
- `shape`
- `reverse`
- `reverse_sequence`
- `sliding_windows`

### 3.6 Reshape / layout / structure

- `batch_to_space`
- `concat`
- `depth_to_space`
- `expand_dims`
- `pixel_shuffle`
- `pixel_unshuffle`
- `reshape`
- `reshape_like`
- `space_to_batch`
- `space_to_depth`
- `split`
- `squeeze`
- `stack`
- `transpose`
- `slice_by_index`
- `slice_by_size`

### 3.7 Constant / constexpr / quantization-related ops

- `const`
- `constexpr_sparse_to_dense`
- `constexpr_affine_dequantize`
- `constexpr_lut_to_dense`
- `constexpr_blockwise_shift_scale`
- `constexpr_sparse_blockwise_shift_scale`
- `constexpr_lut_to_sparse`
- `dequantize`
- `quantize`
- `constexpr_cast `

### 3.8 Control flow / state / list ops

- `call`
- `cond`
- `while_loop`
- `slice_update`
- `write_state`
- `read_state`
- `make_list`
- `list_length`
- `list_read`
- `list_gather`
- `list_write`
- `list_scatter`

### 3.9 Random / reduction / sorting / misc

- `random_uniform`
- `random_bernoulli`
- `random_categorical`
- `random_normal`
- `reduce_argmax`
- `reduce_argmin`
- `reduce_l1_norm`
- `reduce_l2_norm`
- `reduce_log_sum`
- `reduce_log_sum_exp`
- `reduce_max`
- `reduce_mean`
- `reduce_min`
- `reduce_prod`
- `reduce_sum`
- `reduce_sum_square`
- `argsort`
- `topk`
- `fill`
- `fill_like`
- `flatten2d`
- `identity`
- `non_maximum_suppression`
- `pad`
- `range_1d`
- `tile`
- `band_part`
- `cumsum`
- `non_zero`
- `one_hot`
- `matrix_decomposition`
- `cross_product`
- `gamma`
- `degamma`

### 3.10 ANE / PE / custom engine-facing ops

- `scaled_dot_product_attention`
- `ne_conv`
- `ne_matmul`
- `ne_pool`
- `ne_bypass`
- `pe_pool`
- `pe_elementwise`
- `pe_goc`

## 4. What this means

This registry tells us several important things.

### 4.1 Apple has bespoke handling for a very broad MIL op surface

This is not a tiny whitelist of primitive ops. The compiler has explicit
handlers for:

- control flow
- state ops
- list ops
- geometry/indexing ops
- buffer-boundary ops
- quantization/constexpr ops
- attention
- PE/NE-specific ops

That means `ValidateOpList(...)` is sitting on top of a substantial semantic
lowering vocabulary, not just validating a handful of common neural-net ops.

### 4.2 Operation handling is name-driven

Because the registry is keyed by string names like:

- `scaled_dot_product_attention`
- `write_state`
- `tensor_to_circular_buffer`

it strongly suggests op handling at this stage is organized around stable MIL op
identifiers, not just type IDs or ad hoc pattern matching.

This matches the earlier observation that failure to recover an operation
identifier is fatal to validation.

### 4.3 `rustane` is operating close to the real vocabulary

One especially important entry is:

- `scaled_dot_product_attention`

That means Apple’s converted-MIL validator has a dedicated handler for the same
logical op family `rustane` has been hand-assembling and debugging around.

Likewise:

- `matmul`
- `einsum`
- `concat`
- `reshape`
- `transpose`
- `slice_by_index`

are all present as first-class handled ops.

That reinforces that the repo is working near the real compiler vocabulary, not
inventing an alien representation.

## 5. What this changes in our understanding

### 5.1 `ValidateOpList(...)` is table-driven over a large hard-coded registry

We can now say more precisely:

- `ValidateOpList(...)` is not just “large”
- it is large because it sits on top of a wide, compiled-in registry of MIL op
  names and handler functions

### 5.2 The remaining validation work is likely split per op family

Because the registry includes many specialized families, the next useful
reverse-engineering steps are probably not one monolithic pass.

They are likely:

- targeted passes on specific op families or helper functions

For example:

- gather / scatter / resize / crop
- state / ring-buffer ops
- attention
- quantization / constexpr ops

### 5.3 The compiler already knows about attention explicitly

The presence of:

- `scaled_dot_product_attention`

is especially important for `rustane`.

It means Apple’s semantic validator does not require attention to be invisible
behind only primitive matmul/softmax/reshape patterns. It has an explicit op
name in the registry.

That does **not** prove the branch should change representation immediately, but
it does mean there is a real Apple-side semantic target if future experiments
want to compare “fused attention op” versus “hand-expanded attention graph”.

## 6. Best next targets from here

Now that we know the registry exists, the best next steps are more specific:

1. one op-family handler from this registry
   - especially `scaled_dot_product_attention`
2. `RevalidateOpIndicesProducer(...)`
   - already known to matter for gather/crop/resize/resample families
3. `ConvertLiveInputs(...)` or `ConvertLiveOutputs(...)`
   - if the focus shifts back to ABI/tensor-info conversion

For `rustane`, the highest-value next target is probably:

- the handler behind `scaled_dot_product_attention`

because it is the clearest direct connection between Apple’s compiled-in MIL
semantic vocabulary and the repo’s current ANE work.
