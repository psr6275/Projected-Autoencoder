from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def projection_DS(model,X,vr=0.9,tol = 10^(-6),max_iter = 1000,viz_image=False,verbose=False,save_image = None):
    """

    :param model:
    :param X:
    :param vr:
    :param tol:
    :param max_iter:
    :param viz_image: the number of steps we want to visualize every
    :param verbose:
    :param save_image: THe path for saving the intermediate images
    :return:
    """
    tempX = X
    revX = model.predict(tempX)
    i=1
    while np.sqrt(np.sum(np.square(revX-tempX)))>tol and i<max_iter:
        prX = vr*tempX+(1-vr)*revX
        tempX = prX
        revX = model.predict(tempX)
        if viz_image:
            if i%viz_image ==1:
                plt.imshow(prX.reshape(28,28))
                plt.show()
                if save_image is not None:
                    plt.savefig(save_image+'_iter_'+str(i)+'.png')
        else:
            if save_image is not None:
                if i%int(max_iter/10)==1:
                    plt.imshow(prX.reshape(28, 28))
                    plt.savefig(save_image + '_iter_' + str(i) + '.png')

        i+=1


    prX = vr*tempX+(1-vr)*revX
    if verbose:
        if i ==max_iter: 
            print("maximum iteration with ",np.sqrt(np.sum(np.square(revX-prX))))
        else:
            print("The difference is:",np.sqrt(np.sum(np.square(revX-prX))))
    return prX
