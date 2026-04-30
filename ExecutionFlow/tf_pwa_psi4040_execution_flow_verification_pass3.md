# Third-Pass Verification of `tf_pwa_psi4040_execution_flow.md`

Target document:

- [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md)

Verification method:

- start from the leaf inputs in [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py), [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\config_a.yml), [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\Resonances.yml), and [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\final_params_full.json)
- trace forward through the actual tf-pwa source
- compare the resulting source path against the execution-flow note

## Confirmed Steps

### 1. Leaf inputs are identified correctly

Confirmed from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_analysis_Gemini.py):

```python
p4_dict = {
    particle_map["D"]: tf.constant(...),
    particle_map["D0"]: tf.constant(...),
    particle_map["K"]: tf.constant(...),
    particle_map["pi"]: tf.constant(...)
}

phsp_variables = config.data.cal_angle(p4_dict)
phsp_variables["c"] = np.array([-1.0])
```

and:

```python
with open(PARAMS_FILE, 'r') as f:
    params_dict = json.load(f)['value']
...
config.set_params(p_unit)
```

Assessment:

- the note now correctly states that `c` is attached after `cal_angle(...)`
- the note correctly treats the hardcoded four-vectors plus YAML/JSON inputs as the leaf sources

### 2. Top-level amplitude-object construction is present

Confirmed from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\config_loader\config_loader.py):

```python
def get_amplitude(self, vm=None, name=""):
    ...
    amp = create_amplitude(...)
    ...
    return amp
```

and from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\amp\amp.py):

```python
def create_amplitude(decay_group, **kwargs):
    ...
    return get_config(AMP_MODEL)[mode](decay_group, **kwargs)
```

Assessment:

- these wrappers were missing in an earlier version
- they are now included and correctly positioned before `DecayGroup.get_amp(...)`

### 3. The kinematic-data branch matches the source path

Confirmed forward path:

- `Data.cal_angle(...)`
- `cal_angle_from_momentum(...)`
- `cal_angle_from_momentum_id_swap(...)`
- `cal_angle_from_momentum_base(...)`
- `cal_angle_from_momentum_single(...)`
- `struct_momentum(...)`
- `infer_momentum(...)`
- `add_mass(...)`
- `cal_angle_from_particle(...)`
- `cal_helicity_angle(...)`
- `aligned_angle_ref_rule1(...)` / `aligned_angle_ref_rule2(...)`
- `add_relative_momentum(...)`

Relevant source excerpt from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\cal_angle.py):

```python
for i in decay_chain_struct:
    data_i = cal_helicity_angle(data, i, base_z=base_z)
    decay_data[i] = data_i
if align_ref == "center_mass":
    set_x, ref_matrix_final = aligned_angle_ref_rule2(...)
else:
    set_x, ref_matrix_final = aligned_angle_ref_rule1(...)
...
add_relative_momentum(data)
```

Assessment:

- the execution-flow note now covers the actual forward wrappers and the missing angle-building layer
- the main physical interpretation in the note is consistent with the code

### 4. The parameter-loading branch matches the source path

Confirmed forward path:

- `ConfigLoader.set_params(...)`
- `AbsPDF.set_params(...)`
- `VarsManager.set_all(...)`
- `VarsManager.set(...)`
- later reads via `Variable.__call__(...)`
- with the actual scalar access performed through `VarsManager.read(...)`

Relevant source excerpt from [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\tf-pwa\tf_pwa\variable.py):

```python
def set_all(self, vals, val_in_fit=False):
    if type(vals) == dict:
        for name in vals:
            self.set(name, vals[name], val_in_fit=val_in_fit)

def set(self, name, value, val_in_fit=True):
    ...
    self.variables[name].assign(value)

def read(self, name):
    val = self.variables[name]
    ...
    return val
```

Assessment:

- the note now separates the write and read layers instead of compressing them
- this matches the source better

### 5. The chain, decay, and particle factors remain correctly described

Confirmed forward path:

- `DecayGroup.get_amp(...)`
- `DecayChain.get_amp(...)`
- `DecayChain.get_amp_particle(...)`
- `HelicityDecay.get_amp(...)`
- `get_helicity_amp(...)`
- `get_ls_amp(...)`
- `get_D_matrix_term(...)`
- `get_relative_momentum2(...)`
- `cache_relative_p2(...)`
- `C(BWR).get_amp(...)`
- underlying `Particle.get_amp(...)`
- `ParticleOne.get_amp(...)`

Assessment:

- the note remains correct on the main amplitude-building path for the `Psi(4040)` component

## Remaining Inaccuracies

No material source-path inaccuracies were found in this pass.

## Missing Steps

No missing steps were found on the intended tf-pwa execution path documented by the note.

There are still deeper mathematical utility layers below the documented level, for example low-level rotation and Lorentz-vector helpers used inside `cal_helicity_angle(...)`. Those are real function calls, but they are below the current document’s practical scope and no longer represent a gap in the tf-pwa execution-flow reconstruction itself.

## Minor Editorial Issues

- section numbering in the execution-flow note has a small duplication:
  - `2.5` is used both for `cal_angle_from_momentum_single(...)` and for the data-branch leaves
  - `3.5` is used both for `VarsManager.read(...)` and for the parameter-branch leaves

This is editorial only. It does not affect technical correctness.

## Final Verdict

The current version of [C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md](C:\Users\gamma\Documents\Playground\B2DxDK.jl_playground\Analysis\tf_pwa_psi4040_execution_flow.md) is technically consistent with the source on the intended execution path from file inputs to the final `Psi(4040)` complex amplitude.

Conclusion:

- no further technical corrections are required for the function-call reconstruction at the current scope
- only minor editorial cleanup remains optional
