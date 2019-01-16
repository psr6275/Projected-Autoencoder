import numpy as np
import matplotlib.pyplot as plt

def draw_plot(imgs,n = None, fig_size = None,rgb =True):
    if n is None:
        n = len(imgs)
    if fig_size is None:
        fig_size = (15,25)
    plt.figure(figsize=fig_size)
    for i in range(n):
        if rgb:
            ax = plt.subplot(n,4,4*i+1)
            plt.imshow(imgs[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(n,4,4*i+2)
            plt.imshow(imgs[i,:,:,0],'gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(n,4,4*i+3)
            plt.imshow(imgs[i,:,:,1],'gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(n,4,4*i+4)
            plt.imshow(imgs[i,:,:,2],'gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax = plt.subplot(1,n,i+1)
            plt.imshow(imgs[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


