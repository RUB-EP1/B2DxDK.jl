from tf_pwa.config_loader import ConfigLoader
import numpy as np

config = ConfigLoader("config_a.yml")

phsp = config.generate_phsp_p(1000000)

config.data.savetxt("../flat_mc.dat", phsp)

