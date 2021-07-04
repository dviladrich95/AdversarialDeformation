import math
from demo_mnist import demo_mnist

sigma_begin = 7
sigma_end = 8
batch_size = 3
plot_defs = False
# 'affine' translate
adversarial = 'clean'
norm_type = 'elastic'

l_list = [ math.pow(2,i) for i in range(sigma_begin,sigma_end)]
mu_list = [ math.pow(2,i) for i in range(sigma_begin,sigma_end)]
sigma_list = [ math.pow(2,i) for i in range(sigma_begin,sigma_end)]

for sigma,l,mu in zip(sigma_list,l_list,mu_list):
    demo_mnist('clean',batch_size,sigma,1, 1, 'inf_norm', plot_defs)
    demo_mnist('translate',batch_size,sigma,1, 1, 'inf_norm', plot_defs)
    #demo_mnist(adversarial, batch_size, 1, 1, mu, 'elastic', plot_defs)
    #demo_mnist(adversarial, batch_size, 1, l, 1, 'elastic', plot_defs)