 
# examples of custom particle model
from tf_pwa.amp import simple_resonance, AmpDecayChain, register_decay_chain, get_particle_model
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table
import tensorflow as tf
# import custom_1p_model
# import psi_kmatrix


@simple_resonance("New", params=["alpha", "beta"])
def New_Particle(m, alpha, beta):
    """example Particle model define, can be used in config.yml as `model: New`"""
    zeros = tf.zeros_like(m)
    r = -tf.complex(alpha, beta) * tf.complex(m**2 - 4.35**2, zeros)
    return tf.exp(r)


@register_decay_chain("NR1")
class NRDecayChain(AmpDecayChain):
    def init_params(self, name=""):
        self.aD = self.add_var(name+"aD", is_complex=True)
        self.aK = self.add_var(name+"aK", is_complex=True)

    def get_amp(self, data_c, data_p, **kwargs):
        order_map = {str(i): i for i in self.outs}
        p1 = data_p[order_map["D"]]["p"]
        p2 = data_p[order_map["K"]]["p"]
        p3 = data_p[order_map["D0"]]["p"]
        p4 = data_p[order_map["pi"]]["p"]
        pDst = p3 + p4
        pK = p2
        pD = p1
        eta = tf.constant([1, -1, -1,-1], dtype=tf.float64)
        ldot = lambda x, y: tf.reduce_sum(eta * x * y, axis=-1)
        
        mDst = ldot(pDst, pDst)
        
        a1 = - ldot(pD, p3) + ldot(pD, pDst)*ldot(p3, pDst)/mDst
        a2 = - ldot(pK, p3) + ldot(pK, pDst)*ldot(p3, pDst)/mDst
        zeros = tf.zeros_like(a1)
        ret = self.aD() * tf.complex(a1, zeros) 
        ret += self.aK() * tf.complex(a2, zeros) 
        return tf.reshape(ret, (-1, 1,1,1,1,1))

@register_decay_chain("NR_D")
class NRDecayChain(AmpDecayChain):
    def init_params(self, name=""):
        self.aD = self.add_var(name+"aD", is_complex=True)

    def get_angle_amp(self, data_c, data_p, **kwargs):
        order_map = {str(i): i for i in self.outs}
        p1 = data_p[order_map["D"]]["p"]
        p2 = data_p[order_map["K"]]["p"]
        p3 = data_p[order_map["D0"]]["p"]
        p4 = data_p[order_map["pi"]]["p"]
        pDst = p3 + p4
        pK = p2
        pD = p1
        eta = tf.constant([1, -1, -1,-1], dtype=tf.float64)
        ldot = lambda x, y: tf.reduce_sum(eta * x * y, axis=-1)
        
        mDst = ldot(pDst, pDst)
        
        a1 = - ldot(pD, p3) + ldot(pD, pDst)*ldot(p3, pDst)/mDst
        zeros = tf.zeros_like(a1)
        return tf.reshape(tf.complex(a1, zeros), (-1, 1,1,1,1,1))

    def get_m_dep(self, data_c, data_p, **kwargs):
        ret = self.aD()
        zeros = tf.zeros_like(data_p[self.outs[0]]["m"])
        return [tf.reshape(ret+tf.complex(zeros, zeros), (-1,1,1,1,1,1))]


    def get_amp(self, data_c, data_p, **kwargs):
        return self.get_m_dep(data_c, data_p, **kwargs)[0] * self.get_angle_amp(data_c, data_p, **kwargs)

@register_decay_chain("NR_K")
class NRDecayChain(AmpDecayChain):
    def init_params(self, name=""):
        self.aK = self.add_var(name+"aK", is_complex=True)

    def get_angle_amp(self, data_c, data_p, **kwargs):
        order_map = {str(i): i for i in self.outs}
        p1 = data_p[order_map["D"]]["p"]
        p2 = data_p[order_map["K"]]["p"]
        p3 = data_p[order_map["D0"]]["p"]
        p4 = data_p[order_map["pi"]]["p"]
        pDst = p3 + p4
        pK = p2
        pD = p1
        eta = tf.constant([1, -1, -1,-1], dtype=tf.float64)
        ldot = lambda x, y: tf.reduce_sum(eta * x * y, axis=-1)
        
        mDst = ldot(pDst, pDst)

        a2 = - ldot(pK, p3) + ldot(pK, pDst)*ldot(p3, pDst)/mDst
        zeros = tf.zeros_like(a2)
        return tf.reshape(tf.complex(a2, zeros), (-1, 1,1,1,1,1))

    def get_m_dep(self, data_c, data_p, **kwargs):
        ret = self.aK()
        zeros = tf.zeros_like(data_p[self.outs[0]]["m"])
        return [tf.reshape(ret+tf.complex(zeros, zeros), (-1,1,1,1,1,1))]

    def get_amp(self, data_c, data_p, **kwargs):
        return self.get_m_dep(data_c, data_p, **kwargs)[0] * self.get_angle_amp(data_c, data_p, **kwargs)


from tf_pwa.amp.core import register_particle, get_particle_model

def create_particle(name):
    cls = get_particle_model(name)

    @register_particle("C({})".format(name))
    class _NewClass(cls):
        def get_amp(self, data, data_d, all_data=None, **kwargs):
            # print(all_data.keys())
            d = all_data["c"]
            c = getattr(self, "C", 1)
            if c == -1:
                return super().get_amp(data, data_d, all_data=None, **kwargs)
            else:
                amp = super().get_amp(data, data_d, all_data=None, **kwargs)
                return tf.where(d >0, amp, -amp)

    @register_particle("C2({})".format(name))
    class _NewClass(cls):
        def get_amp(self, data, data_d, all_data=None, **kwargs):
            # print(all_data.keys())
            d = all_data["c"]
            amp =  super().get_amp(data, data_d, all_data=None, **kwargs)
            return tf.where(d > 0, tf.zeros_like(amp), amp)

    @register_particle("C3({})".format(name))
    class _NewClass(cls):
        def get_amp(self, data, data_d, all_data=None, **kwargs):
            # print(all_data.keys())
            d = all_data["c"]
            amp =  super().get_amp(data, data_d, all_data=None, **kwargs)
            return tf.where(d < 0, tf.zeros_like(amp), amp)

create_particle("BWR_LS")
create_particle("BWR")
create_particle("BW")
create_particle("one")
create_particle("New")
        
