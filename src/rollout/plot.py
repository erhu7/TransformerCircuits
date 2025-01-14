import numpy as np
import matplotlib.pyplot as plt 

def plot_activation(activation, samp, ax):
    ax.imshow(activation.squeeze(0).detach().numpy())
    samp = samp.squeeze(0).detach().numpy()
    n = len(samp)
    ax.set_xticks(np.arange(0, n), labels=samp)
    ax.set_yticks(np.arange(0, n), labels=samp)
    return ax 