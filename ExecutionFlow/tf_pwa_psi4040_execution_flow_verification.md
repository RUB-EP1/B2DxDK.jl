# Independent Verification of `tf_pwa_psi4040_execution_flow.md`

Target document:

- [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md)

Verification scope:

- Start from the actual input leaves only:
  - hardcoded `p4_dict` and `c` in [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py)
  - model inputs from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)
  - resonance definitions from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml)
  - parameter values from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- Trace forward through the source until the `Psi(4040)` complex amplitude is produced
- Compare that forward trace against the existing markdown

## Forward Trace Used for Verification

### 1. Input leaves in the script

Source excerpt from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py):

```python
with open(PARAMS_FILE, 'r') as f:
    params_dict = json.load(f)['value']

p4_dict = {
    particle_map["D"]: tf.constant([[2.0452, -0.1467, 0.2235, -0.7847]], dtype=tf.float64),
    particle_map["D0"]: tf.constant([[2.2606, 0.2284, -0.3689, 1.2019]], dtype=tf.float64),
    particle_map["K"]: tf.constant([[0.7718, -0.0873, 0.1803, -0.5584]], dtype=tf.float64),
    particle_map["pi"]: tf.constant([[0.2017, 0.0056, -0.0349, 0.1413]], dtype=tf.float64)
}

phsp_variables = config.data.cal_angle(p4_dict)
phsp_variables["c"] = np.array([-1.0])
```

Verified leaf inputs:

- four-vectors come directly from the script, not from `event_vectors.json`
- `c` is assigned directly after `cal_angle(...)`, not passed into `cal_angle(...)`
- `params_dict` comes from `final_params_full.json["value"]`

### 2. Runtime setup for the `Psi(4040)` chain

Independent runtime check of `dg.chains[5]`:

```text
[Bp->Psi(4040)+K, Psi(4040)->Dst+D, Dst->D0+pi]
0 Bp->Psi(4040)+K      ls=((1, 1),)
1 Psi(4040)->Dst+D     ls=((1, 1),)
2 Dst->D0+pi           ls=((1, 0),)
inner=[Dst, Psi(4040)]
```

This confirms:

- chain index `5` is the `Psi(4040)` chain
- the three decay steps are exactly the ones described in the original note
- the chain carries two inner-particle factors: `Dst` and `Psi(4040)`

### 3. Parameter-loading path

Forward path from the file-backed parameters:

1. script builds `p_unit`
2. `ConfigLoader.set_params(...)`
3. `ConfigLoader.get_amplitude(...)`
4. `create_amplitude(...)`
5. `AbsPDF.set_params(...)`
6. `VarsManager.set_all(...)`
7. later reads happen through `Variable.__call__()`

Relevant source excerpts:

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py):

```python
def get_amplitude(self, vm=None, name=""):
    ...
    amp = create_amplitude(decay_group, vm=vm, name=name, ...)
    ...
    return amp

def set_params(self, params, neglect_params=None):
    ...
    amplitude = self.get_amplitude()
    ...
    amplitude.set_params(ret)
```

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py):

```python
def create_amplitude(decay_group, **kwargs):
    ...
    return get_config(AMP_MODEL)[mode](decay_group, **kwargs)

def set_params(self, var):
    self.vm.set_all(var)
```

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py):

```python
def set_all(self, vals, val_in_fit=False):
    if type(vals) == dict:
        for name in vals:
            self.set(name, vals[name], val_in_fit=val_in_fit)
```

### 4. Kinematic-data path

Forward path from the hardcoded four-vectors:

1. `config.data.cal_angle(p4_dict)`
2. `self.preprocessor(...)`
3. `cal_angle_from_momentum(...)`
4. `cal_angle_from_momentum_id_swap(...)`
5. `cal_angle_from_momentum_base(...)`
6. `cal_angle_from_momentum_single(...)`
7. helpers:
   - `struct_momentum(...)`
   - `infer_momentum(...)`
   - `add_mass(...)`
   - `cal_angle_from_particle(...)`
   - `add_relative_momentum(...)`

Relevant source excerpts:

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\data.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\data.py):

```python
def cal_angle(self, p4, data_type="data", **kwargs):
    ...
    data = self.preprocessor(
        {"p4": p4, "extra": kwargs}, data_type=data_type
    )
    return data
```

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py):

```python
def cal_angle_from_momentum(...):
    data = cal_angle_from_momentum_id_swap(...)
    ...
    return data

def cal_angle_from_momentum_single(...):
    data_p = struct_momentum(...)
    ...
    data_p = infer_momentum(data_p, dec)
    data_p = add_mass(data_p, dec)
    data_d = cal_angle_from_particle(...)
    data = {"particle": data_p, "decay": data_d}
    add_relative_momentum(data)
    return CalAngleData(data)
```

### 5. Final amplitude path

Forward path from the prepared data and parameters:

1. `dg.set_used_chains([5])`
2. `dg.get_amp(phsp_variables)`
3. `DecayGroup.get_amp(...)`
4. `DecayChain.get_amp(...)`
5. for each decay step: `HelicityDecay.get_amp(...)`
6. for inner particles: `DecayChain.get_amp_particle(...)`
7. `Psi(4040)` factor via `C(BWR).get_amp(...)`
8. `Dst` factor via `ParticleOne.get_amp(...)`
9. script extracts scalar with `.numpy().flatten()[0]`

Relevant source excerpts:

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py):

```python
def get_amp(self, data):
    ...
    amp = decay_chain.get_amp(data_c, data_p, base_map=base_map, all_data=data)
    ...
    ret = tf.reduce_sum(ret, axis=0)
    return ret
```

```python
def get_amp(self, data_c, data_p, all_data=None, base_map=None):
    ...
    amp_d.append(i.get_amp(data_c[i], data_p, all_data=all_data))
    ...
    rs = self.get_amp_particle(data_p, data_c, all_data=all_data)
    total = self.get_cp_amp_total(...)
    ...
    ret = einsum(idx_s, *amp_d)
    return ret
```

```python
def get_amp(self, data, data_p, **kwargs):
    D_conj = self.get_D_matrix_term(data, data_p, **kwargs)
    H = self.get_helicity_amp(data, data_p, **kwargs)
    ...
    return ret
```

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\extra_amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\extra_amp.py):

```python
def get_amp(self, data, data_d, all_data=None, **kwargs):
    d = all_data["c"]
    c = getattr(self, "C", 1)
    if c == -1:
        return super().get_amp(data, data_d, all_data=None, **kwargs)
    ...
```

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py):

```python
def get_amp(self, data, data_c, **kwargs):
    mass = self.get_mass()
    width = self.get_width()
    ...
    ret = BWR(...) or BW(...)
    return ret
```

From [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\base.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\base.py):

```python
@regist_particle("one")
class ParticleOne(Particle):
    def get_amp(self, data, _data_c=None, **kwargs):
        ...
        return tf.complex(ones, zeros)
```

## Confirmed Steps

The original markdown is correct on these points:

- The target chain is `dg.chains[5]`, and it is `[Bp->Psi(4040)+K, Psi(4040)->Dst+D, Dst->D0+pi]`.
- The final script call of interest is `dg.get_amp(phsp_variables).numpy().flatten()[0]`.
- The kinematic branch is centered on `config.data.cal_angle(...)` and ultimately builds `particle`, `decay`, and `|q|^2` data.
- The parameter branch is centered on `ConfigLoader.set_params(...)`, `AbsPDF.set_params(...)`, `VarsManager.set_all(...)`, and `Variable.__call__()`.
- `DecayGroup.get_amp(...)` delegates to `DecayChain.get_amp(...)`.
- `DecayChain.get_amp(...)` combines:
  - decay-step tensors
  - chain normalization `total`
  - inner-particle factors
- `HelicityDecay.get_amp(...)`, `get_helicity_amp(...)`, `get_ls_amp(...)`, and `get_D_matrix_term(...)` are all on the real forward path.
- The `Psi(4040)` particle factor is `C(BWR)` and the `Dst` particle factor is `one`.
- The `Psi(4040)` model definition in the input files is consistent with the note:
  - `Psi(4040)` is in the `DstD` channel
  - `model: C(BWR)`
  - `C: -1`

## Missing Steps in the Original Markdown

These functions do execute in the forward path but were omitted from the original note:

1. `ConfigLoader.get_amplitude(...)`
   - This is the actual bridge from YAML-decay configuration to the runtime amplitude object.

2. `create_amplitude(...)`
   - The amplitude instance is not implicit; it is explicitly created here.

3. `cal_angle_from_momentum_id_swap(...)`
   - This wrapper runs even when there are no identical-particle swaps to perform.

4. `cal_angle_from_momentum_base(...)`
   - `cal_angle_from_momentum(...)` does not jump directly to `cal_angle_from_momentum_single(...)`; the base wrapper is in the real call path.

5. The final script-side scalar extraction
   - `dg.get_amp(...)` returns a tensor-like amplitude object; the scalar complex number used for comparison is only obtained after `.numpy().flatten()[0]`.

## Extra or Overstated Items in the Original Markdown

These points are not wrong in spirit, but they are broader than the strict forward execution path:

1. The original note presents `cal_angle_from_particle(...)` as a single conceptual step.
   - That is acceptable for overview purposes, but it compresses several internal angle-building helpers.

2. The original note treats the parameter branch as stopping at `ConfigLoader.set_params(...) -> amplitude.set_params(...) -> vm.set_all(...)`.
   - That is functionally true for parameter loading, but it skips the separate amplitude-creation path, which also depends on the YAML inputs.

## Inaccuracies

### 1. `c` is not passed into `cal_angle(...)` in this script

The original note says, in the `Data.cal_angle` section, that the function takes optional extra variables such as `c`. That is true for the API in general, but it is not what this script actually does.

Actual script behavior:

```python
phsp_variables = config.data.cal_angle(p4_dict)
phsp_variables["c"] = np.array([-1.0])
```

Consequence:

- `c` is added after `cal_angle(...)` returns
- the verification report therefore treats `c` as a separate leaf input, not as an argument consumed by `Data.cal_angle(...)` in this execution

### 2. Minor path compression in the data branch

The original note says `cal_angle_from_momentum(...)` is the next relevant function after `Data.cal_angle(...)`. That is directionally correct, but the literal forward path contains the intermediate wrappers:

- `cal_angle_from_momentum_id_swap(...)`
- `cal_angle_from_momentum_base(...)`

This is an omission rather than a conceptual error, but it matters for a strict layer-by-layer reconstruction.

## Final Verdict

Verdict: **substantially correct, but incomplete as a strict forward execution trace**

Assessment:

- The original markdown correctly captures the main physics objects, the correct `Psi(4040)` chain, the correct decay/resonance factors, and the main argument-production branches.
- The core interpretation of the returned complex amplitude is sound.
- For a strict source-level verification, the note is missing several real intermediary functions and contains one concrete inaccuracy: `c` is not supplied through `cal_angle(...)` in this script.

Recommended reading order:

1. Use the original markdown for the conceptual overview.
2. Use this verification report for the exact forward-execution corrections and missing wrappers.
