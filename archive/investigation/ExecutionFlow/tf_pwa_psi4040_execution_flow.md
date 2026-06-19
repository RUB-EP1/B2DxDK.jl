# Psi(4040) Complex-Amplitude Execution Flow in `tf_pwa_analysis_Gemini.py`

This note reconstructs the execution flow that produces the complex amplitude for the `Psi(4040)` contribution in [tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py).

Scope:

- Target component: `("Psi(4040) [L=1, l=1]", 5, 0, 0)` from `granular_groups`
- Target call: `dg.get_amp(phsp_variables)`
- Goal: recursively reconstruct all functions needed to produce the arguments and internal state used by that call, until only file-backed or hardcoded inputs remain

## 0. Script-Level Setup for the `Psi(4040)` Probe

### 0.1 The script selects the `Psi(4040)` chain and evaluates `dg.get_amp`

Source: [tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py)

```python
amp_model = config.get_amplitude()
dg = amp_model.decay_group
all_chains = dg.chains

granular_groups = [
    ...
    ("Psi(4040) [L=1, l=1]", 5, 0, 0),
    ...
]

for name, chain_idx, prod_ls, decay_ls in granular_groups:
    dg.set_used_chains([chain_idx])
    chain = all_chains[chain_idx]
    ...
    config.set_params(p_unit)
    val = dg.get_amp(phsp_variables).numpy().flatten()[0]
```

Arguments needed here:

- `phsp_variables`: kinematic data dictionary
- internal amplitude parameters already loaded into `config` via `config.set_params(p_unit)`
- chain-selection state set by `dg.set_used_chains([5])`

Physical meaning:

- `val` is the complex partial-wave amplitude of the selected `Psi(4040)` chain at one specific phase-space point.
- The script chooses only one chain and one LS component, so the returned number is not the full decay amplitude, but one isolated building block of the coherent model.

### 0.2 The script constructs the kinematic argument `phsp_variables`

```python
p4_dict = {
    particle_map["D"]: tf.constant([[2.0452, -0.1467, 0.2235, -0.7847]], dtype=tf.float64),
    particle_map["D0"]: tf.constant([[2.2606, 0.2284, -0.3689, 1.2019]], dtype=tf.float64),
    particle_map["K"]: tf.constant([[0.7718, -0.0873, 0.1803, -0.5584]], dtype=tf.float64),
    particle_map["pi"]: tf.constant([[0.2017, 0.0056, -0.0349, 0.1413]], dtype=tf.float64)
}

phsp_variables = config.data.cal_angle(p4_dict)
phsp_variables["c"] = np.array([-1.0])
```

Arguments needed:

- four-vectors for `D`, `D0`, `K`, `pi`

Physical meaning:

- these four-vectors define one event in phase space
- `cal_angle(...)` transforms them into the invariant masses, breakup momenta, and helicity/aligned angles needed by the amplitude model
- `c = -1` is attached to `phsp_variables` after `cal_angle(...)` returns, and then selects the negative branch of the custom `C(...)` particle wrappers from [extra_amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\extra_amp.py)

Leaf inputs:

- hardcoded four-vectors in the script
- particle names and ordering from [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)

### 0.3 The script constructs the parameter state `p_unit`

```python
with open(PARAMS_FILE, 'r') as f:
    params_dict = json.load(f)['value']

params_ones = params_dict.copy()
for k in params_ones:
    if "total" in k or "g_ls" in k:
        if k.endswith("r"):
            params_ones[k] = 1.0
        elif k.endswith("i"):
            params_ones[k] = 0.0

...

p_zero = params_dict.copy()
for k in p_zero:
    if "total" in k or "g_ls" in k:
        if k.endswith("r") or k.endswith("i"):
            p_zero[k] = 0.0

p_unit = p_zero
for d_idx, d in enumerate(chain.chain):
    ...
    for k in p_unit:
        if prefix in k and ("total" in k or "g_ls" in k):
            if k.endswith(f"_{ls_idx}r"):
                p_unit[k] = 1.0

config.set_params(p_unit)
```

Arguments needed:

- `params_dict` from [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- chain structure from `all_chains[5]`

Physical meaning:

- all fitted couplings are first zeroed
- then only the `Psi(4040)` chain normalization and the selected LS couplings are set to `1 + 0i`
- this isolates the pure kinematic and line-shape contribution of the chosen partial wave

Leaf inputs:

- fitted parameter names and values from [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- model topology from [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)

Relevant `Psi(4040)` file-backed parameters:

- `Psi(4040)_mass = 4.039`
- `Psi(4040)_width = 0.08`
- `Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0r`
- `Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_0i`
- `Bp->Psi(4040).K_g_ls_0r`
- `Bp->Psi(4040).K_g_ls_0i`
- `Psi(4040)->Dst.D_g_ls_0r`
- `Psi(4040)->Dst.D_g_ls_0i`
- `Dst->D0.pi_g_ls_0r`
- `Dst->D0.pi_g_ls_0i`

## 1. Top-Level Function: `DecayGroup.get_amp(data)`

Before `DecayGroup.get_amp(data)` is available, the script first builds the amplitude object through the configuration layer.

### 1.1 `ConfigLoader.get_amplitude(vm=None, name="")`

Source: [tf_pwa/config_loader/config_loader.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py)

```python
def get_amplitude(self, vm=None, name=""):
    amp_config = self.config.get("data", {})
    ...
    decay_group = self.full_decay
    ...
    amp = create_amplitude(
        decay_group,
        vm=vm,
        name=name,
        use_tf_function=use_tf_function,
        no_id_cached=no_id_cached,
        jit_compile=jit_compile,
        model=amp_model,
        cached_shape_idx=cached_shape_idx,
        all_config=amp_config,
    )
    self.add_constraints(amp)
    self.amps[vm] = amp
    return amp
```

Arguments:

- the configured full decay group
- optional variable manager and configuration flags

Physical meaning:

- constructs the runtime amplitude object from the YAML-defined decay topology and particle models

### 1.2 `create_amplitude(decay_group, **kwargs)`

Source: [tf_pwa/amp/amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py)

```python
def create_amplitude(decay_group, **kwargs):
    mode = kwargs.get("model", "default")
    ...
    return get_config(AMP_MODEL)[mode](decay_group, **kwargs)
```

Arguments:

- `decay_group`
- amplitude-model mode and configuration options

Physical meaning:

- instantiates the concrete tf-pwa amplitude model whose `decay_group` is later queried with `get_amp(data)`

Source: [tf_pwa/amp/core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py)

```python
def get_amp(self, data):
    data_particle = data["particle"]
    data_decay = data["decay"]

    used_chains = tuple([self.chains[i] for i in self.chains_idx])
    chain_maps = self.get_chains_map(used_chains)
    base_map = self.get_base_map()
    ret = []
    for chains in chain_maps:
        for decay_chain in chains:
            chain_topo = decay_chain.standard_topology()
            ...
            data_c = rename_data_dict(data_decay_i, chains[decay_chain])
            data_p = rename_data_dict(data_particle, chains[decay_chain])
            amp = decay_chain.get_amp(
                data_c, data_p, base_map=base_map, all_data=data
            )
            ret.append(amp)
    ret = tf.reduce_sum(ret, axis=0)
    return ret
```

Arguments:

- `data["particle"]`: particle four-vectors and masses for final and intermediate states
- `data["decay"]`: helicity-angle data keyed by decay topology

Physical meaning:

- this sums the complex amplitudes of all currently enabled decay chains
- in the `Psi(4040)` probe, `self.chains_idx = [5]`, so the sum contains only the `Psi(4040)` chain

Functions that must have run beforehand:

1. `dg.set_used_chains([5])`
2. `config.data.cal_angle(p4_dict)` to build `data`
3. `config.set_params(p_unit)` to load the selected couplings and resonance parameters

## 2. Data Branch: How `phsp_variables` Is Produced

### 2.1 `Data.cal_angle(p4, **kwargs)`

Source: [tf_pwa/config_loader/data.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\data.py)

```python
def cal_angle(self, p4, data_type="data", **kwargs):
    if isinstance(p4, (list, tuple)):
        p4 = {k: v for k, v in zip(self.get_dat_order(), p4)}
    ...
    data = self.preprocessor(
        {"p4": p4, "extra": kwargs}, data_type=data_type
    )
    return data
```

Arguments:

- `p4`: dictionary of final-state four-vectors
- optional extra variables such as `c`

Physical meaning:

- converts raw event kinematics into the angle/mass data structure expected by the amplitude model

### 2.2 `cal_angle_from_momentum(p, decs, ...)`

Source: [tf_pwa/cal_angle.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py)

```python
def cal_angle_from_momentum(
    p, decs, using_topology=True, center_mass=False,
    r_boost=True, random_z=False, batch=65000,
    align_ref=None, only_left_angle=False
):
    data = cal_angle_from_momentum_id_swap(
        p, decs, using_topology, center_mass, r_boost,
        random_z, batch, align_ref=align_ref,
        only_left_angle=only_left_angle,
    )
    ...
    return data
```

Arguments:

- `p`: final-state four-vectors
- `decs`: full decay topology from the configuration

Physical meaning:

- dispatches the momentum-to-angle conversion for the configured decay group

### 2.3 `cal_angle_from_momentum_id_swap(p, decs, ...)`

Source: [tf_pwa/cal_angle.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py)

```python
def cal_angle_from_momentum_id_swap(
    p, decs, using_topology=True, center_mass=False,
    r_boost=True, random_z=False, batch=65000,
    align_ref=None, only_left_angle=False,
):
    data = cal_angle_from_momentum_base(
        p, decs, using_topology, center_mass, r_boost,
        random_z, batch, align_ref=align_ref,
        only_left_angle=only_left_angle,
    )
    if id_particles is None or len(id_particles) == 0:
        return data
    ...
```

Arguments:

- `p`: final-state four-vectors
- `decs`: decay group

Physical meaning:

- adds the symmetry-handling layer for identical particles before returning the kinematic data
- in this specific model there are no identical final-state particles, but this wrapper still lies on the forward call path

### 2.4 `cal_angle_from_momentum_base(p, decs, ...)`

Source: [tf_pwa/cal_angle.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py)

```python
def cal_angle_from_momentum_base(
    p, decs, using_topology=True, center_mass=False,
    r_boost=True, random_z=False, batch=65000,
    align_ref=None, only_left_angle=False,
):
    if data_shape(p) is None:
        return cal_angle_from_momentum_single(
            p, decs, using_topology, center_mass, r_boost,
            random_z, align_ref=align_ref,
            only_left_angle=only_left_angle,
        )
    ret = []
    for i in split_generator(p, batch):
        ret.append(
            cal_angle_from_momentum_single(
                i, decs, using_topology, center_mass, r_boost,
                random_z, align_ref=align_ref,
                only_left_angle=only_left_angle,
            )
        )
    return data_merge(*ret)
```

Arguments:

- `p`: final-state four-vectors
- `decs`: decay group

Physical meaning:

- batching wrapper around the single-event/single-batch momentum-to-angle conversion
- for the one-event probe in this script, it forwards into `cal_angle_from_momentum_single(...)`

### 2.5 `cal_angle_from_momentum_single(p, decs, ...)`

```python
def cal_angle_from_momentum_single(...):
    p = {BaseParticle(k) if isinstance(k, str) else k: v for k, v in p.items()}
    p = {i: p[i] for i in decs.outs}
    data_p = struct_momentum(p, center_mass=center_mass)
    if using_topology:
        decay_chain_struct = decs.topology_structure()
    else:
        decay_chain_struct = decs
    for dec in decay_chain_struct:
        data_p = infer_momentum(data_p, dec)
        data_p = add_mass(data_p, dec)
    data_d = cal_angle_from_particle(
        data_p, decs, using_topology,
        r_boost=r_boost, random_z=random_z,
        align_ref=align_ref, only_left_angle=only_left_angle,
    )
    data = {"particle": data_p, "decay": data_d}
    add_relative_momentum(data)
    return CalAngleData(data)
```

Arguments:

- final-state four-vectors
- decay topology

Physical meaning:

- this is the main constructor of the `phsp_variables` object passed into `dg.get_amp`
- it creates:
  - `data["particle"]`: four-vectors and invariant masses of all particles and intermediate systems
  - `data["decay"]`: helicity and aligned angles for every decay step in every topology
  - `|q|^2` values for each two-body decay

### 2.6 Helper functions called by `cal_angle_from_momentum_single`

#### `struct_momentum`

```python
def struct_momentum(p, center_mass=True):
    ret = {}
    ...
    for i in p:
        ret[i] = {"p": p[i]}
    return ret
```

Physical meaning:

- wraps each external four-vector into the internal data structure `{particle: {"p": p4}}`

#### `infer_momentum`

```python
def infer_momentum(data, decay_chain):
    st = decay_chain.sorted_table()
    for i in st:
        if i in data:
            continue
        ps = []
        for j in st[i]:
            ps.append(data[j]["p"])
        data[i] = {"p": tf.reduce_sum(ps, 0)}
    return data
```

Physical meaning:

- reconstructs intermediate-state four-vectors, for example the `Dst`, the `Psi(4040)` candidate, and the mother `Bp`, by summing daughters

#### `add_mass`

```python
def add_mass(data, _decay_chain=None):
    for i in data:
        if isinstance(i, BaseParticle):
            p = data[i]["p"]
            data[i]["m"] = LorentzVector.M(p)
    return data
```

Physical meaning:

- computes invariant masses from the reconstructed four-vectors

#### `cal_angle_from_particle`

```python
def cal_angle_from_particle(
    data, decay_group, using_topology=True,
    random_z=True, r_boost=True, final_rest=True,
    align_ref=None, only_left_angle=False,
):
    ...
    decay_data = {}
    ...
```

Physical meaning:

- computes the helicity Euler angles and aligned angles used in the Wigner-D parts of the amplitude

#### `cal_helicity_angle`

```python
def cal_helicity_angle(
    data, decay_chain, base_z=np.array([0.0, 0.0, 1.0]),
    base_x=np.array([1.0, 0.0, 0.0]),
):
    ret = {}
    part_data = cal_chain_boost(data, decay_chain)
    set_x = {decay_chain.top: base_x}
    set_z = {decay_chain.top: base_z}
    ...
    for j in i.outs:
        ...
        ang, x = EulerAngle.angle_zx_z_getx(
            set_z[i.core], set_x[i.core], z2
        )
        ...
        ret[i][j]["ang"] = ang
        ret[i][j]["x"] = x
        ret[i][j]["z"] = z2
    ...
    return ret
```

Physical meaning:

- computes the actual helicity Euler angles for each decay step by boosting daughters into the mother rest frame and comparing their directions to the chosen reference axes
- these `ang` values are later consumed by `get_D_matrix_term(...)`

#### Alignment-reference helpers

```python
def aligned_angle_ref_rule1(decay_group, decay_chain_struct, decay_data, data):
    ...
    return set_x, ref_matrix_final

def aligned_angle_ref_rule2(decay_group, decay_chain_struct, decay_data, data):
    ...
    return set_x, ref_matrix_final
```

Physical meaning:

- choose the reference axes used to define aligned angles across different decay chains
- these helpers are part of the angle-construction layer underneath `cal_angle_from_particle(...)`
- they matter because the chain tensors are later rotated into a common spin basis before interference is summed

#### `add_relative_momentum`

```python
def add_relative_momentum(data):
    data_p = data["particle"]
    for decay_chain in data["decay"]:
        for decay in decay_chain:
            m0 = data_p[decay.core]["m"]
            m1 = data_p[decay.outs[0]]["m"]
            m2 = data_p[decay.outs[1]]["m"]
            p2 = Getp2(m0, m1, m2)
            data["decay"][decay_chain][decay]["|q|2"] = p2
    return data
```

Physical meaning:

- stores the two-body breakup momentum squared for each decay step
- these values feed barrier factors and running widths later

### 2.7 Leaves of the data branch

No further function-derived inputs are needed after this point. The leaf inputs are:

- hardcoded four-vectors in [tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py)
- particle identities and decay topology from [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)

## 3. Parameter Branch: How `config.set_params(p_unit)` Loads the `Psi(4040)` Couplings

### 3.1 `ConfigLoader.set_params(params, neglect_params=None)`

Source: [tf_pwa/config_loader/config_loader.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py)

```python
def set_params(self, params, neglect_params=None):
    ...
    amplitude = self.get_amplitude()
    ret = params.copy()
    ...
    amplitude.set_params(ret)
    return True
```

Arguments:

- `params`: here the script-built dictionary `p_unit`

Physical meaning:

- transfers named parameter values into the live amplitude object

### 3.2 `AbsPDF.set_params(var)`

Source: [tf_pwa/amp/amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py)

```python
def set_params(self, var):
    self.vm.set_all(var)
```

Physical meaning:

- forwards the parameter dictionary into the variable manager

### 3.3 `VarsManager.set_all(vals)`

Source: [tf_pwa/variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py)

```python
def set_all(self, vals, val_in_fit=False):
    if type(vals) == dict:
        for name in vals:
            self.set(name, vals[name], val_in_fit=val_in_fit)
```

Physical meaning:

- assigns every named scalar to the internal TensorFlow variables

### 3.4 `VarsManager.set(name, value, val_in_fit=True)`

Source: [tf_pwa/variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py)

```python
def set(self, name, value, val_in_fit=True):
    if val_in_fit and name in self.bnd_dic:
        value = self.bnd_dic[name].get_x2y(value)
    if name in self.variables:
        if name in self.pre_trans:
            trans = self.pre_trans[name]
            value = trans.inverse(value)
        if value is not None:
            self.variables[name].assign(value)
```

Physical meaning:

- performs the actual per-variable assignment into the runtime TensorFlow state
- this is the concrete write step for names like `Psi(4040)_mass` or `Bp->Psi(4040).K_g_ls_0r`

### 3.5 `VarsManager.read(name)`

Source: [tf_pwa/variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py)

```python
def read(self, name):
    val = self.variables[name]
    if name in self.mask_vars:
        val = tf.stop_gradient(tf.cast(self.mask_vars[name], val.dtype))
    if name in self.pre_trans:
        trans = self.pre_trans[name]
        val = trans(self.variables)
    return val
```

Physical meaning:

- retrieves one runtime scalar from the variable manager
- this is the concrete read step used when downstream objects ask for masses, widths, chain totals, and LS couplings

### 3.6 `Variable.__call__`

```python
def __call__(self, charge=1):
    var = [self.vm.read(i) for i in self.all_name_list]
    ...
    ret_rect = tf.complex(r, i)
    ret = tf.where(cond, ret_polar, ret_rect)
    return tf.reshape(ret, self.shape)
```

Physical meaning:

- later, when a decay or particle asks for `g_ls()`, `total()`, `mass()`, or `width()`, this returns the live value that was loaded from the JSON or overwritten by the script

### 3.7 Leaves of the parameter branch

No further function-derived inputs are needed after this point. The leaf inputs are:

- model definitions from [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)
- resonance definitions from [Resonances.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml)
- fitted parameter values from [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- the script-side override that sets only the selected `Psi(4040)` LS amplitudes to `1 + 0i`

## 4. Recursive Expansion of `dg.get_amp` for the `Psi(4040)` Chain

With only chain `5` enabled, `DecayGroup.get_amp(...)` calls the `Psi(4040)` chain object.

### 4.1 `DecayChain.get_amp(data_c, data_p, all_data=None, base_map=None)`

Source: [tf_pwa/amp/core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py)

```python
def get_amp(self, data_c, data_p, all_data=None, base_map=None):
    ...
    for i in self:
        indices.append(i.amp_index(base_map))
        amp_d.append(i.get_amp(data_c[i], data_p, all_data=all_data))

    if self.need_amp_particle:
        rs = self.get_amp_particle(data_p, data_c, all_data=all_data)
        total = self.get_cp_amp_total(
            charge=all_data.get("charge_conjugation", 1)
        )
        if rs is not None:
            total = total * tf.cast(rs, total.dtype)
        amp_d.append(total)
        indices.append([])
    ...
    ret = einsum(idx_s, *amp_d)
    return ret
```

Arguments:

- `data_c`: decay-angle data for this chain
- `data_p`: particle masses and four-vectors
- `all_data`: whole event dictionary, including `c`

Physical meaning:

- multiplies the sequential decay tensors for
  - `Bp -> Psi(4040) + K`
  - `Psi(4040) -> Dst + D`
  - `Dst -> D0 + pi`
- multiplies them by the chain normalization `total`
- multiplies by particle factors such as the `Psi(4040)` line shape and the `Dst` factor
- contracts all spin/helicity indices into one complex chain amplitude

### 4.2 `DecayChain.get_amp_particle(data_p, data_c, all_data=None)`

```python
def get_amp_particle(self, data_p, data_c, all_data=None):
    amp_p = []
    ...
    for i in self.inner:
        ...
        amp_p.append(i.get_amp(data_p[i], data_c_i, all_data=all_data))
    rs = 1.0
    for i in amp_p:
        rs = rs * i
    return rs
```

Physical meaning:

- multiplies the mass-dependent particle factors of the intermediate resonances in the chain
- for the `Psi(4040)` chain this includes:
  - the `Psi(4040)` resonance factor
  - the `Dst` factor

## 5. Decay-Step Tensor Factors

Each decay step uses `HelicityDecay.get_amp`.

### 5.1 `HelicityDecay.get_amp(data, data_p, **kwargs)`

Source: [tf_pwa/amp/core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py)

```python
def get_amp(self, data, data_p, **kwargs):
    D_conj = self.get_D_matrix_term(data, data_p, **kwargs)
    H = self.get_helicity_amp(data, data_p, **kwargs)
    H = tf.reshape(H, (-1, 1, *self.n_helicity_inner()))
    H = tf.cast(H, dtype=D_conj.dtype)
    ret = H * tf.stop_gradient(D_conj)
    self.add_algin(ret, data)
    return ret
```

Arguments:

- `data`: the angular data for one decay step
- `data_p`: particle masses and four-vectors

Physical meaning:

- combines:
  - the LS/helicity coupling structure `H`
  - the Wigner-D rotation factor `D_conj`
- this is the basic spin-amplitude tensor for one two-body decay

### 5.2 `get_helicity_amp`

```python
def get_helicity_amp(self, data, data_p, **kwargs):
    m_dep = self.get_ls_amp(data, data_p, **kwargs)
    cg_trans = tf.cast(self.get_cg_matrix(), m_dep.dtype)
    n_ls = len(self.get_ls_list())
    m_dep = tf.reshape(m_dep, (-1, n_ls, 1, 1))
    cg_trans = tf.reshape(cg_trans, (n_ls, *self.n_helicity_inner()))
    H = tf.reduce_sum(m_dep * cg_trans, axis=1)
    ...
    return tf.reshape(H, (-1, 1, *self.n_helicity_inner()))
```

Physical meaning:

- transforms LS amplitudes into helicity amplitudes using Clebsch-Gordan coefficients

### 5.3 `get_ls_amp`

```python
def get_ls_amp(self, data, data_p, **kwargs):
    g_ls = self.get_g_ls()
    q0 = self.get_relative_momentum2(data_p, False)
    data["|q0|2"] = q0
    q = self.cache_relative_p2(data, data_p)
    if self.has_barrier_factor:
        bf = self.get_barrier_factor2(
            data_p[self.core]["m"], q, q0, self.d
        )
        m_dep = g_ls * tf.cast(to_complex(bf), g_ls.dtype)
    else:
        m_dep = tf.reshape(g_ls, (1, -1))
    return m_dep
```

Arguments:

- decay-step data and particle data

Physical meaning:

- combines the complex LS couplings `g_ls` with Blatt-Weisskopf-type barrier factors
- for the `Psi(4040)` probe, only one LS component is left nonzero by the script

### 5.4 `get_D_matrix_term`

```python
def get_D_matrix_term(self, data, data_p, **kwargs):
    a = self.core
    b = self.outs[0]
    c = self.outs[1]
    ang = data[b]["ang"]
    D_conj = get_D_matrix_lambda(
        ang, a.J, a.spins, *self.list_helicity_inner()
    )
    ...
    return D_conj
```

Physical meaning:

- builds the Wigner-D angular factor from the helicity Euler angles computed in the data branch

### 5.5 `get_relative_momentum2` and `cache_relative_p2`

```python
def get_relative_momentum2(self, data, from_data=False):
    ...
    m0 = _get_mass(self.core)
    m1 = _get_mass(self.outs[0])
    m2 = _get_mass(self.outs[1])
    ret = get_relative_p2(m0, m1, m2)
    return ret

def cache_relative_p2(self, data, data_p):
    if "|q|2" in data:
        q = data["|q|2"]
    else:
        q = self.get_relative_momentum2(data_p, True)
        data["|q|2"] = q
    return q
```

Physical meaning:

- computes the breakup momentum squared of the decay in the mother rest frame
- feeds barrier factors and running-width terms

Leaves for the decay-step branch:

- LS couplings from [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- masses from `data_p` and the YAML particle definitions
- angles from `cal_angle(...)`

## 6. Particle Factors for the `Psi(4040)` Chain

### 6.1 Custom wrapper `C(BWR)` from `extra_amp.py`

Source: [extra_amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\extra_amp.py)

```python
@register_particle("C({})".format(name))
class _NewClass(cls):
    def get_amp(self, data, data_d, all_data=None, **kwargs):
        d = all_data["c"]
        c = getattr(self, "C", 1)
        if c == -1:
            return super().get_amp(data, data_d, all_data=None, **kwargs)
        else:
            amp = super().get_amp(data, data_d, all_data=None, **kwargs)
            return tf.where(d > 0, amp, -amp)
```

Arguments:

- `data`: resonance mass data
- `data_d`: breakup momentum data
- `all_data["c"]`: the custom sign-control variable

Physical meaning:

- wraps the ordinary particle model with a sign choice controlled by the model attribute `C` and the event-level variable `c`
- in [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml), `Psi(4040)` uses `model: C(BWR)` and `C: -1`, so for this state the wrapper passes through the underlying Breit-Wigner unchanged

### 6.2 Underlying Breit-Wigner `Particle.get_amp`

Source: [tf_pwa/amp/core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py)

```python
def get_amp(self, data, data_c, **kwargs):
    mass = self.get_mass()
    width = self.get_width()
    if width is None:
        return tf.ones_like(data["m"])
    if not self.running_width:
        ret = BW(data["m"], mass, width)
    else:
        q = data_c["|q|"]
        q0 = data_c["|q0|"]
        ...
        ret = BWR(data["m"], mass, width, q, q0, self.bw_l, self.d)
    ...
    return ret
```

Arguments:

- `data["m"]`: invariant mass of the resonance candidate
- `data_c["|q|"]`, `data_c["|q0|"]`: breakup momenta
- internal parameters `mass`, `width`

Physical meaning:

- computes the complex resonance propagator for `Psi(4040)`
- for `Psi(4040)` in this model, the mass and width come from the JSON/YAML parameter set

Leaf inputs:

- `Psi(4040)_mass`
- `Psi(4040)_width`
- breakup momenta from the data branch

### 6.3 `Dst` factor: model `one`

Source: [tf_pwa/amp/base.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\base.py)

```python
@regist_particle("one")
class ParticleOne(Particle):
    def init_params(self):
        pass

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = data["m"]
        zeros = tf.zeros_like(mass)
        ones = tf.ones_like(mass)
        return tf.complex(ones, zeros)
```

Physical meaning:

- the `Dst` particle contributes a trivial factor `1 + 0i`
- all nontrivial `Dst` dependence therefore comes from the `Dst -> D0 + pi` decay tensor, not from a resonance line shape

Leaf input:

- model choice `Dst: model: one` from [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)

## 7. Final Recursive Summary for `Psi(4040)`

The complex amplitude returned by:

```python
val = dg.get_amp(phsp_variables).numpy().flatten()[0]
```

for the `Psi(4040)` component is built from:

1. Event kinematics
   - hardcoded `D`, `D0`, `K`, `pi` four-vectors in [tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py)
   - converted by `cal_angle(...)` into masses, helicity angles, aligned angles, and `|q|^2`

2. Model topology and quantum numbers
   - [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)
   - [Resonances.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml)

3. Runtime-selected parameter state
   - loaded from [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
   - modified by the script so only the selected `Psi(4040)` chain normalization and LS couplings remain nonzero, each set to `1 + 0i`

4. Chain-internal factors
   - decay tensors from `Bp -> Psi(4040) K`, `Psi(4040) -> Dst D`, `Dst -> D0 pi`
   - chain normalization `total`
   - `Psi(4040)` particle factor from `C(BWR)`
   - trivial `Dst` particle factor from `one`

So the recursive reconstruction stops at these non-function leaf values:

- four-vectors in the script
- `c = -1` in the script
- particle and decay definitions in [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)
- resonance definitions in [Resonances.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml)
- fitted parameter values in [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
