import math

import numpy as np

from demo_mnist import demo_mnist

sigma_begin = 6
sigma_end = 7
batch_size = 1000
plot_defs = False
# 'affine' translate
adversarial = 'clean'
# 'inf_norm'
norm_type = 'elastic'

l_list = [ math.pow(2,i) for i in range(sigma_begin,sigma_end)]
mu_list = [ math.pow(2,i) for i in range(sigma_begin,sigma_end)]
sigma_list = [ math.pow(2,i) for i in range(sigma_begin,sigma_end)]

for sigma,l,mu in zip(sigma_list,l_list,mu_list):
    #how smoothing affects success
    #demo_mnist('clean',batch_size,sigma,1, 1, 'elastic', plot_defs,max_norm=np.inf)

    #how l, mu affects deformation type
    #demo_mnist('clean',batch_size,1,l, 0, 'elastic', plot_defs, max_norm=100)
    #demo_mnist('clean',batch_size,1,0, mu, 'elastic', plot_defs,max_norm=100)
    #demo_mnist('clean',batch_size,1,l, mu, 'elastic', plot_defs,max_norm=20)

    #how smoothing affects translation trained nets
    #demo_mnist('translate',batch_size,sigma,1, 1, 'inf_norm', plot_defs, max_norm=np.inf)

    #demo_mnist('translate',batch_size,sigma,1, 1, 'inf_norm', plot_defs, max_norm=3.0)
    #demo_mnist('clean', batch_size, sigma, 1, 1, 'inf_norm', plot_defs, max_norm=3.0)
    #demo_mnist(adversarial, batch_size, 1, 1, mu, 'elastic', plot_defs)
    #demo_mnist(adversarial, batch_size, 1, l, 1, 'elastic', plot_defs)