# Second-Pass Verification of `tf_pwa_psi4040_execution_flow.md`

Target document:
[C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md)

Verification scope:

- Start from true leaf inputs only:
  - hardcoded four-vectors and `c` handling in [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py)
  - file-backed model and parameter inputs from:
    - [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml)
    - [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml)
    - [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- Trace forward to the final complex-amplitude return path for the `Psi(4040)` component.

## Confirmed Steps

### 1. Leaf inputs are identified correctly

Confirmed from [tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py):

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

Confirmed from [final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json):

- `Psi(4040)_mass`
- `Psi(4040)_width`
- `Bp->Psi(4040).KPsi(4040)->Dst.DDst->D0.pi_total_*`
- `Bp->Psi(4040).K_g_ls_*`
- `Psi(4040)->Dst.D_g_ls_*`
- `Dst->D0.pi_g_ls_*`

Confirmed from [config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml):

- `DstD` contains `Psi(4040)`
- `Psi(4040)` uses `model: C(BWR)`
- `Dst` uses `model: one`

### 2. The strict top-level construction path is now present

Confirmed from [tf_pwa/config_loader/config_loader.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py):

```python
def get_amplitude(self, vm=None, name=""):
    ...
    amp = create_amplitude(decay_group, ...)
    ...
    return amp
```

Confirmed from [tf_pwa/amp/amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py):

```python
def create_amplitude(decay_group, **kwargs):
    mode = kwargs.get("model", "default")
    return get_config(AMP_MODEL)[mode](decay_group, **kwargs)
```

The updated note now includes both wrappers. This matches the forward source path.

### 3. The `c` handling is now described correctly

The updated note states that `c = -1` is attached after `cal_angle(...)` returns. That matches the source exactly.

### 4. The main kinematic path is correctly described

Confirmed path:

1. `config.data.cal_angle(p4_dict)`
2. `cal_angle_from_momentum(...)`
3. `cal_angle_from_momentum_id_swap(...)`
4. `cal_angle_from_momentum_base(...)`
5. `cal_angle_from_momentum_single(...)`
6. `struct_momentum(...)`
7. `infer_momentum(...)`
8. `add_mass(...)`
9. `cal_angle_from_particle(...)`
10. `add_relative_momentum(...)`

The updated note now includes items 2-5 explicitly. This is materially correct.

### 5. The main parameter-loading path is correctly described

Confirmed path:

1. script builds `p_unit`
2. `ConfigLoader.set_params(...)`
3. `get_amplitude()`
4. `AbsPDF.set_params(...)`
5. `VarsManager.set_all(...)`
6. later reads happen via `Variable.__call__`

Source excerpts:

From [tf_pwa/config_loader/config_loader.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py):

```python
def set_params(self, params, neglect_params=None):
    ...
    amplitude = self.get_amplitude()
    ...
    amplitude.set_params(ret)
```

From [tf_pwa/amp/amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py):

```python
def set_params(self, var):
    self.vm.set_all(var)
```

From [tf_pwa/variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py):

```python
def set_all(self, vals, val_in_fit=False):
    if type(vals) == dict:
        for name in vals:
            self.set(name, vals[name], val_in_fit=val_in_fit)
```

### 6. The main amplitude path is correctly described

Confirmed from [tf_pwa/amp/core.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\core.py):

```python
def get_amp(self, data):
    ...
    amp = decay_chain.get_amp(data_c, data_p, base_map=base_map, all_data=data)
    ...
    ret = tf.reduce_sum(ret, axis=0)
    return ret
```

Then:

- `DecayChain.get_amp(...)`
- `DecayChain.get_amp_particle(...)`
- `HelicityDecay.get_amp(...)`
- `get_helicity_amp(...)`
- `get_ls_amp(...)`
- `get_D_matrix_term(...)`
- `Particle.get_amp(...)` for `Psi(4040)`
- `ParticleOne.get_amp(...)` for `Dst`

This matches the updated note on the main physics path.

## Remaining Inaccuracies

### 1. The data branch still stops one layer early inside `cal_angle_from_particle(...)`

The note lists `cal_angle_from_particle(...)`, but it does not recursively expand the real helper functions that actually build the angles:

- `cal_helicity_angle(...)`
- `aligned_angle_ref_rule1(...)` or `aligned_angle_ref_rule2(...)`

Source from [tf_pwa/cal_angle.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py):

```python
for i in decay_chain_struct:
    data_i = cal_helicity_angle(data, i, base_z=base_z)
    decay_data[i] = data_i
...
set_x, ref_matrix_final = aligned_angle_ref_rule1(...)
```

Impact:

- The note is correct at a high level, but it is not yet fully recursive on the kinematic-angle sub-branch.

### 2. The parameter branch compresses real intermediary function layers

The note lists:

- `VarsManager.set_all(...)`
- `Variable.__call__(...)`

But the actual forward and later retrieval path includes at least:

- `VarsManager.set(...)`
- `VarsManager.read(...)`

Source from [tf_pwa/variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py):

```python
def set_all(self, vals, val_in_fit=False):
    ...
    self.set(name, vals[name], val_in_fit=val_in_fit)

def read(self, name):
    val = self.variables[name]
    ...
    return val
```

Impact:

- The note still compresses the variable-manager path rather than giving it strictly layer by layer.

## Missing Steps

These steps are present in the actual forward trace but not expanded as separate steps in the updated note:

1. `cal_helicity_angle(...)`
2. `aligned_angle_ref_rule1(...)` / `aligned_angle_ref_rule2(...)`
3. `VarsManager.set(...)`
4. `VarsManager.read(...)`

These are not cosmetic omissions. They are real functions on the forward path from leaves to final amplitude.

## Extra or Unsupported Claims

No major unsupported physics claims were found in the updated note.

The following statements are acceptable:

- `c` is attached after `cal_angle(...)`
- `Psi(4040)` uses `C(BWR)` and effectively passes through to the underlying Breit-Wigner because `C = -1`
- `Dst` contributes a trivial `one` particle factor

## Final Verdict

Verdict: mostly correct, but still not fully recursive enough to satisfy the original requirement literally.

What is correct:

- the leaf inputs
- the top-level wrapper path
- the main data path down to `cal_angle_from_particle(...)`
- the main parameter-loading path
- the chain, decay, and particle-factor structure

What still remains incomplete:

- the kinematic-angle branch is not expanded through `cal_helicity_angle(...)` and alignment helpers
- the parameter branch compresses `set_all -> set` and `read -> Variable.__call__`

Practical assessment:

- As a physics/software overview, the note is good.
- As a strict layer-by-layer recursive reconstruction from leaves to the final amplitude, it is still incomplete.

