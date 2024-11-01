
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes=(100, 200, 300,400)):
        super(BinaryClassifier, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layer_sizes)):
            layers.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_dim))
        # Sigmoid activation for binary classification output
        layers.append(nn.Sigmoid())

        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def pitch_angle(ep,R,Z):
    return np.squeeze(np.arctan(ep.BpFunc(R,Z)/np.abs(ep.BtFunc(R,Z)))*180./np.pi)

def efit_params(ep):
    Ip = ep.g['Ip']
    Bt0 = ep.g['Bt0']
    if 'psiN' in ep.g.keys():
        psin = ep.g['psiN']
    else:
        psin = ep.g['psi']
    q95 = np.interp(0.95, psin,ep.g['q'])
    Rc1 = 1.575; Zc1 = -1.30
    Rc2 = 1.72; Zc2 = -1.51
    alpha1 = pitch_angle(ep, Rc1, Zc1)
    alpha2 = pitch_angle(ep, Rc2, Zc2)

    return Bt0, Ip, q95, alpha1, alpha2


class AIShadow():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        f = torch.load('model_aishadow.pth', map_location=self.device)
        self.indices = f['indices'] #indices the NN predicts for (fixed CAD and mesh)
        self.input_norm = pickle.loads(f['scaler_pkl']) #
        self.model = BinaryClassifier(input_dim=f['input_dim'], output_dim=f['output_dim'])
        self.model.load_state_dict(f['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def inference(self, ep):
        _, Ip, q95, alpha1, alpha2 = efit_params(ep)
        print("EFIT predictions: %0.2f, %0.2f, %0.2f, %0.2f" % (Ip, q95, alpha1, alpha2))
        x = np.zeros((1, 4)) #singleton for number of samples
        x[0,:] = np.array([Ip, q95, alpha1, alpha2])

        #RMC - maybe add a check here that Zlowest is < -0.9 (where we limited training data)

        #normalize input data
        xnorm = self.input_norm.transform(x)
        xnorm = torch.tensor(xnorm, dtype=torch.float32)
        #create full array TODO: don't hardcode
        ypredict = np.ones((498559,)) 
        #inference with AI model of points that change
        with torch.no_grad():
            ypredict[self.indices] = np.round(self.model(xnorm).numpy())

        return ypredict

        

