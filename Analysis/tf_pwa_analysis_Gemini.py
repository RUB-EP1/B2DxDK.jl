import numpy as np
import tensorflow as tf
from tf_pwa.config_loader import ConfigLoader 
import json
import extra_amp

# --- Configuration ---
CONFIG_FILE = "config_a.yml"
PARAMS_FILE = "final_params_full.json"

# --- Main Logic ---
try:
    # 1. Load Configuration
    config = ConfigLoader(CONFIG_FILE)
    
    # 2. Load Parameters
    with open(PARAMS_FILE, 'r') as f:
        params_dict = json.load(f)['value']
    
    # Set all couplings to 1+0i for comparison
    params_ones = params_dict.copy()
    for k in params_ones:
        if "total" in k or "g_ls" in k:
            if k.endswith("r"):
                params_ones[k] = 1.0
            elif k.endswith("i"):
                params_ones[k] = 0.0
                
    config.set_params(params_ones) 

    # 3. Define Kinematics (Hardcoded for consistency with Julia)
    # Using the same phase space point as in pure_model.jl
    particles = list(config.get_decay().outs)
    particle_map = {p.name: p for p in particles}
    
    with open("event_vectors.json", 'r') as f:
        vecs_root = json.load(f)
        vecs = vecs_root["vectors"]
    
    # Hardcoded phase space point from Julia comparison

    p4_dict = {
        particle_map["D"]: tf.constant([[2.0452, -0.1467, 0.2235, -0.7847]], dtype=tf.float64),
        particle_map["D0"]: tf.constant([[2.2606, 0.2284, -0.3689, 1.2019]], dtype=tf.float64),
        particle_map["K"]: tf.constant([[0.7718, -0.0873, 0.1803, -0.5584]], dtype=tf.float64),
        particle_map["pi"]: tf.constant([[0.2017, 0.0056, -0.0349, 0.1413]], dtype=tf.float64)
    }

    phsp_variables = config.data.cal_angle(p4_dict)
    phsp_variables["c"] = np.array([-1.0]) # Extra variable required by config
    
    # 4. Calculate Amplitudes
    amp_model = config.get_amplitude()
    dg = amp_model.decay_group
    all_chains = dg.chains
    
    # Define the granular mapping (19 chains to match Julia)
    # Each entry: (res_name, chain_idx, prod_ls_idx, decay_ls_idx)
    
    granular_groups = [
        ("X(3872) [L=1, l=0]", 0, 0, 0),
        ("X(3872) [L=1, l=2]", 0, 0, 1),
        ("X(3915)(0-) [L=1, l=1]", 1, 0, 0),
        ("chi(c2)(3930) [L=1, l=1]", 2, 0, 0),
        ("X(3940)(1.) [L=0, l=0]", 3, 0, 0),
        ("X(3940)(1.) [L=0, l=2]", 3, 0, 1),
        ("X(3993) [L=0, l=0]", 4, 0, 0),
        ("X(3993) [L=0, l=2]", 4, 0, 1),
        ("Psi(4040) [L=1, l=1]", 5, 0, 0),
        ("X(4300) [L=0, l=0]", 6, 0, 0),
        ("X(4300) [L=0, l=2]", 6, 0, 1),
        ("NR(1-)PPm [L=1, l=1]", 10, 0, 0),
        ("NR(0-)SPm [L=0, l=1]", 9, 0, 0),
        ("NR(1+)PSp [L=1, l=0]", 8, 0, 0),
        ("NR(0-)SPp [L=0, l=1]", 7, 0, 0),
        ("X0(2900) [L=0, l=0]", 11, 0, 0),
        ("X1(2900) [L=0, l=1]", 12, 0, 0),
        ("X1(2900) [L=1, l=1]", 12, 1, 0),
        ("X1(2900) [L=2, l=1]", 12, 2, 0)
    ]

    with open("amplitudes.txt", "w") as f:
        f.write("--- tf-pwa Results ---\n")
        
        # 1. Granular chains
        for name, chain_idx, prod_ls, decay_ls in granular_groups:
            dg.set_used_chains([chain_idx])
            chain = all_chains[chain_idx]
            
            # Set all relevant couplings to 0 first
            p_zero = params_dict.copy()
            for k in p_zero:
                if "total" in k or "g_ls" in k:
                    if k.endswith("r") or k.endswith("i"):
                        p_zero[k] = 0.0
            
            # Set units for all steps in the chain
            p_unit = p_zero
            for d_idx, d in enumerate(chain.chain):
                c_name = d.core.name.replace("(1+)", "(1.)")
                o_names = [p.name.replace("(1+)", "(1.)") for p in d.outs]
                prefix = f"{c_name}->{'.'.join(o_names)}"
                
                # Determine which LS index to use
                ls_idx = 0
                if d_idx == 0: ls_idx = prod_ls
                if d_idx == 1: ls_idx = decay_ls
                
                for k in p_unit:
                    if prefix in k and ("total" in k or "g_ls" in k):
                        if k.endswith(f"_{ls_idx}r"): p_unit[k] = 1.0
            
            config.set_params(p_unit)
            val = dg.get_amp(phsp_variables).numpy().flatten()[0]
            
            sign = "+" if val.imag >= 0 else "-"
            line = f"Resonance ({name}): {val.real} {sign} {abs(val.imag)}im\n"
            f.write(line)
            print(line.strip())
            
    print("Exported amplitudes.txt")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nError: {e}")