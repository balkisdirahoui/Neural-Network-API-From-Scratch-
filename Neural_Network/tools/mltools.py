import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan"], [".", "+", "*", "o", "x", "^"]
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], marker="x")
        return
    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels == l, 0], data[labels == l, 1], c=cols[i], marker=marks[i])


def plot_frontiere(data, f, step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), colors=('gray', 'blue'), levels=[-1, 0, 1])


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:, 0]), np.min(data[:, 0]), np.max(data[:, 1]), np.min(data[:, 1])
        xmax, xmin, ymax, ymin = xmax+0.1, xmin-0.1, ymax+0.1, ymin-0.1
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step),
                       np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y


def gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type == 0:
        # melange de 2 gaussiennes
        xpos = np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), nbex // 2)
        xneg = np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), nbex // 2)
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
    if data_type == 1:
        # melange de 4 gaussiennes
        xpos = np.vstack((np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), nbex // 4),
                          np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), nbex // 4)))
        xneg = np.vstack((np.random.multivariate_normal([-centerx, centerx], np.diag([sigma, sigma]), nbex // 4),
                          np.random.multivariate_normal([centerx, -centerx], np.diag([sigma, sigma]), nbex // 4)))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))

    if data_type == 2:
        # echiquier
        data = np.reshape(np.random.uniform(-4, 4, 2 * nbex), (nbex, 2))
        y = np.ceil(data[:, 0]) + np.ceil(data[:, 1])
        y = 2 * (y % 2) - 1
    # un peu de bruit
    data[:, 0] += np.random.normal(0, epsilon, nbex)
    data[:, 1] += np.random.normal(0, epsilon, nbex)
    # on mÃ©lange les donnÃ©es
    idx = np.random.permutation((range(y.size)))
    data = data[idx, :]
    y = y[idx]
    return data, y


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_image(data,n=16):
    plt.imshow(data.reshape((n,n)),interpolation="nearest",cmap="gray")


def plot_img(X_test,y_test,net,nb_pred=6,n=16):
    random_ind = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(15,5*np.ceil(nb_pred / 4)))
    j = 1
    for i in random_ind:
        plt.subplot(np.ceil(nb_pred / 4),4,j)
        plt.title("classe predite : {0} / vraie classe : {1}".format(net.predict(np.asarray([X_test[i]])), y_test[i]))
        show_image(X_test[i],n)
        j+=1



def plot_img_o(X_test,y_test,net,nb_pred=8,n=16):
    random_ind = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(16,7))
    j = 1
    for i in random_ind:
        plt.subplot(3,5,j)
        plt.title("vraie classe : {1}".format(net.predict(np.asarray([X_test[i]])), y_test[i]))
        show_image(X_test[i],n)
        j+=1

      
def draw_construction(X_test,net,nb_pred=None,n=28,n_comp=4):
    
    if nb_pred == None:
        ids = [1,3,5,7,2,0,18,15,17,4]
        nb_pred=10
    else:
        ids = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(5*nb_pred,15))
    j = 1
    for i in ids:
        plt.subplot(3,nb_pred,j)
        plt.title("Real")
        show_image(X_test[i],n)
        
        plt.subplot(3,nb_pred,j+nb_pred)
        plt.title("Compressed")
        show_image(net.forward(np.asarray([X_test[i]]))[4],n_comp)
        
        plt.subplot(3,nb_pred,j+2*nb_pred)
        plt.title("Reconstructed")
        show_image(net.predict(np.asarray([X_test[i]])),n)
        j+=1

def add_noise(data,type="gaussian",p=0.1):
    
    if type == "gaussian":
        return data + p * np.random.normal(loc=0.0, scale=0.5, size=data.shape) 
    if type == "salt_pepper":
        out = data + np.random.choice([0, 1], size=data.shape, p=[1-p, p])
        return np.where(out > 1,1,out)
    else:
        print("wrong type")

def draw_noise(X_test,X_bruit,net,nb_pred=None,n=28,n_comp=4):
    
    if nb_pred == None:
        ids = [1,3,5,7,2,0,18,15,17,4]
        nb_pred=10
    else:
        ids = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(5*nb_pred,15))
    j = 1
    for i in ids:
        plt.subplot(3,nb_pred,j)
        plt.title("Real")
        show_image(X_test[i],n)
        
        plt.subplot(3,nb_pred,j+nb_pred)
        plt.title("with Noise")
        show_image(X_bruit[i],n)
        
        plt.subplot(3,nb_pred,j+2*nb_pred)
        plt.title("Reconstructed")
        show_image(net.predict(np.asarray([X_bruit[i]])),n)
        j+=1

#####################################################
import numpy as np
import copy 

 

class Sequentiel:
    
    def __init__(self, modules,labels=None):
        #s'assurer qu'il ya assez de module (plus que 1)
        assert(len(modules) > 0)
        self._modules = modules
        self._labels = labels

    def forward(self, x):
        l = [x]
        #ajouter des modules en serie
        for module in self._modules:
            l.append(module.forward(l[-1]))
        l.reverse()
        return l
        

    def backward(self, s, delta):
        #backward 
        list_ = [delta]
        for i, module in enumerate(np.flip(self._modules)):
            module.backward_update_gradient(s[i+1], list_[-1])
            list_.append(module.backward_delta(s[i+1] , list_[-1]))
        return list_
    
    def update_parameters(self, eps = 1e-3):
        for module in self._modules:
            module.update_parameters(gradient_step=eps)
            module.zero_grad()

            
    
    def predict(self, x):
        if self._labels is not None:
            return self._labels(self.forward(x)[0])
        return self.forward(x)[0]
    

class Optim:
    
    def __init__(self, net, loss, eps = 1e-3):
        self._net  = net
        self._loss = loss
        self._eps  = eps
    # step(batch_x,batch_y) qui calcule la sortie du réseau 
    #sur batch_x, calcule le coût par rapport aux labels batch_y,
    def step(self,batch_x, batch_y):
        
        s = self._net.forward(batch_x)
        loss = self._loss.forward(batch_y,s[0])
        delta = self._loss.backward(batch_y, s[0])
        self._net.backward(s, delta)
        self._net.update_parameters(self._eps)
        
        return loss
        
    def SGD(self, input_x,lab , batch_size, epochs=10,earlystop=100):
        n = len(input_x)
        len_y = len(lab)
        assert n == len_y
        
        i = np.random.permutation(len(input_x))
        input_x = input_x[i]
        lab = lab[i]
        #batch
        batch_in  = [input_x[i:i + batch_size] for i in range(0, n, batch_size)]
        batch_lab = [lab[i:i + batch_size] for i in range(0, len_y, batch_size)]
        #MEAN,std
        hist_mean = []
        hist_std = []
        #best
        loss_min=float("inf")
        best_ep = 0
        stop=0
        best_net = self._net
        for epoch in range(epochs):
            l = []
            for x,y in zip(batch_in, batch_lab):
                l.append(np.asarray(self.step(x, y)).mean())
            l = np.asarray(l)
            loss = l.mean()
            stop+=1
            if(loss < loss_min):
                stop=0
                best_ep = epoch
                loss_min = loss
                best_net = copy.deepcopy(self._net)
            
            if stop == earlystop:
                break
            hist_mean.append(loss)
            hist_std.append(l.std())
        self._net = best_net
        return hist_mean, hist_std
                
    def score(self,x,y):
        return np.where(y == self._net.predict(x),1,0).mean()
def params(n,d,dat):

   if dat == 0:
     parameters = np.random.random((n, d)) - 0.5
   elif dat == 1:
     parameters = np.random.normal(0, 1, (n, d))
   else :
     parameters = np.random.normal(0, 1, (n, d)) * np.sqrt(2 / (n + d))
   return parameters

def biais_params(n,d,dat):
      if dat == 0:
          biais = np.random.random((1, d)) - 0.5
      elif dat== 1:
          biais = np.random.normal(0, 1, (1, d))
      else :
          biais = np.random.normal(0, 1, (1, d)) * np.sqrt(2 / (n + d))
      return biais