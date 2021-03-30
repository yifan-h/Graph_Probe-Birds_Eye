import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


## sub-function descriptions
## please refer to main py file for function descriptions
'''
binary_classifer: PyTorch model for link prediction (with different hidden layers)
graph_probe: PyTorch model for probing
'''


class binary_classifer(nn.Module):
    def __init__(self,
                layers_num=5, 
                feat_dim=0,
                hidden_dim=128):
        super(binary_classifer, self).__init__()
        self.layers_num = layers_num
        self.linear_sh = nn.Linear(feat_dim, hidden_dim)
        self.linear_dh = nn.Linear(feat_dim, hidden_dim)
        self.linear_h1 = nn.Linear(2* hidden_dim, hidden_dim)
        self.linear_h2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_h3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_h4 = nn.Linear(hidden_dim, hidden_dim)
        if layers_num == 0:  # 0
            self.linear_ca = nn.Linear(2*feat_dim, 1)
        elif layers_num <= 1:  # 1
            self.linear_ca = nn.Linear(2*hidden_dim, 1)
        else:  # 2, 3, 4, 5
            self.linear_ca = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear_ca.weight.data)

    def forward(self, src, dst):
        # layers_num = 0
        h = torch.cat((src, dst), dim=1)
        if self.layers_num == 0:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 1
        s = F.relu(self.linear_sh(src))
        d = F.relu(self.linear_dh(dst))
        h = torch.cat((s, d), dim=1)
        if self.layers_num == 1:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 2
        h = F.relu(self.linear_h1(h))
        if self.layers_num == 2:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 3
        h = F.relu(self.linear_h2(h))
        if self.layers_num == 3:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 4
        h = F.relu(self.linear_h3(h))
        if self.layers_num == 4:
            return torch.sigmoid(self.linear_ca(h))

        # layers_num = 5
        h = F.relu(self.linear_h4(h))
        if self.layers_num == 5:
            return torch.sigmoid(self.linear_ca(h))


class graph_probe(nn.Module):
    def __init__(self, 
                adj_dim,
                feat_dim,
                hidden_dim=64):
        super(graph_probe, self).__init__()
        self.linear_ah = nn.Linear(adj_dim, hidden_dim)
        self.linear_fh = nn.Linear(feat_dim, hidden_dim)
        self.linear_hh2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.linear_hh3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_hh4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_hh5 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear_hm = nn.Linear(hidden_dim*2, 1)
        nn.init.xavier_normal_(self.linear_ah.weight.data)
        nn.init.xavier_normal_(self.linear_fh.weight.data)
        nn.init.xavier_normal_(self.linear_hm.weight.data)

    def forward(self, a, f):
        a = self.linear_ah(a)
        f = self.linear_fh(f)
        h = torch.cat((f, a), dim=1)
        # h = F.elu(self.linear_hh2(h))
        # h = F.elu(self.linear_hh3(h))
        # h = F.elu(self.linear_hh4(h))
        # h = F.elu(self.linear_hh5(h))
        return F.elu(self.linear_hm(h))
        # return self.linear_hm(h)


'''
class autoencoder(nn.Module):
    def __init__(self, 
                feat_dim,
                hidden_dim=8):
        super(autoencoder, self).__init__()
        self.linear_fh = nn.Linear(feat_dim, hidden_dim)
        self.linear_hf = nn.Linear(hidden_dim, feat_dim)
        nn.init.xavier_normal_(self.linear_fh.weight.data)
        nn.init.xavier_normal_(self.linear_hf.weight.data)

    def forward(self, f, encoding=False):
        h = F.elu(self.linear_fh(f))
        if encoding: 
            return h.detach().cpu().numpy()
        else:
            return F.elu(self.linear_hf(h))

class graph_probe_attn(nn.Module):
    def __init__(self, 
                 adj_dim,
                 feat_dim,
                 hidden_dim=64,
                 d_model=96, 
                 dropout=0.1):
        super(graph_probe_attn, self).__init__()
        self.dropout = dropout
        self.linear_context = nn.Linear(adj_dim, d_model)
        self.linear_query = nn.Linear(feat_dim, d_model)
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, d_model)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)
        bias = torch.empty(1)
        self.bias = nn.Parameter(bias)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        nn.init.constant_(bias, 0)

        self.linear_oh = nn.Linear(d_model*4, hidden_dim)
        self.linear_hh = nn.Linear(hidden_dim, hidden_dim)
        self.linear_hm = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear_oh.weight.data)
        nn.init.xavier_normal_(self.linear_hh.weight.data)
        nn.init.xavier_normal_(self.linear_hm.weight.data)

    def forward(self, a, f):
        # context-query attention
        C = self.linear_context(a)  # context=graph
        Q = self.linear_query(f)  # query=feature
        Lc, d_model = C.shape
        Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        S1 = F.softmax(S, dim=1)
        S2 = F.softmax(S, dim=0)
        A = torch.matmul(S1, Q)
        B = torch.matmul(torch.matmul(S1, S2.transpose(0, 1)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=1)

        h = F.elu(self.linear_oh(out))
        h = F.elu(self.linear_hh(h))
        return F.elu(self.linear_hm(h))


    def trilinear_for_attention(self, C, Q):
        Lc, d_model = C.shape
        Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(0, 1).expand([Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(0,1))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


# MINEE
def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

def _normal_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    # return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min
    return torch.randn((batch_size, data_min.shape[0]))

def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

class MINEE():
    class Net(nn.Module):
        # Inner class that defines the neural network architecture
        def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)

        def forward(self, inputs):

            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            return output

    def __init__(self, x_dim, y_dim, ref_batch_factor=1, lr=1e-4, hidden_size=100):
        self.lr = lr
        self.ref_batch_factor = ref_batch_factor
        self.XY_net = MINEE.Net(input_size=x_dim + y_dim, hidden_size=100)
        self.X_net = MINEE.Net(input_size=x_dim, hidden_size=100)
        self.Y_net = MINEE.Net(input_size=y_dim, hidden_size=100)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(), lr=lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(), lr=lr)

    def step(self, X, Y, iter=1):
        r"""Train the neural networks for one or more steps.
        Argument:
        iter (int, optional): number of steps to train.
        """
        self.X = X
        self.Y = Y
        self.batch_size = X.shape[0]
        self.XY = torch.cat((self.X, self.Y), dim=1)
        for i in range(iter):
            self.XY_optimizer.zero_grad()
            self.X_optimizer.zero_grad()
            self.Y_optimizer.zero_grad()
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_X = _resample(self.X, batch_size=self.batch_size)
            batch_Y = _resample(self.Y, batch_size=self.batch_size)
            batch_X_ref = _normal_sample(self.X, batch_size=int(
                                self.ref_batch_factor * self.batch_size))
            batch_Y_ref = _normal_sample(self.Y, batch_size=int(
                                self.ref_batch_factor * self.batch_size))
            batch_XY_ref = torch.cat((batch_X_ref, batch_Y_ref), dim=1)

            batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref)
            batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref)
            batch_loss_Y = -_div(self.Y_net, batch_Y, batch_Y_ref)

            val_loss_XY = batch_loss_XY.data.item()
            val_loss_X = batch_loss_X.data.item()
            val_loss_Y = batch_loss_Y.data.item()
            val_loss_sum = val_loss_XY + val_loss_X + val_loss_Y

            if val_loss_sum != 0:
                batch_loss_XY = (1 - val_loss_XY/val_loss_sum) * batch_loss_XY
                batch_loss_X = (1 - val_loss_X/val_loss_sum) * batch_loss_X
                batch_loss_Y = (1 - val_loss_Y/val_loss_sum) * batch_loss_Y

            total_loss = batch_loss_XY + batch_loss_X + batch_loss_Y
            total_loss.backward()
            self.XY_optimizer.step()
            self.X_optimizer.step()
            self.Y_optimizer.step()

            return batch_loss_XY.data.item(), batch_loss_X.data.item(), \
                                                    batch_loss_Y.data.item()

    def forward(self, X, Y):
        r"""Evaluate the neural networks to return an array of 3 divergences estimates 
        (dXY, dX, dY).
        Outputs:
            dXY: divergence of sample joint distribution of (X,Y) 
                to the uniform reference
            dX: divergence of sample marginal distribution of X 
                to the uniform reference
            dY: divergence of sample marginal distribution of Y
                to the uniform reference
        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        XY = torch.cat((X, Y), dim=1)
        X_ref = _normal_sample(X, batch_size=int(self.ref_batch_factor * X.shape[0]))
        Y_ref = _normal_sample(Y, batch_size=int(self.ref_batch_factor * Y.shape[0]))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)

        ce_XY = _div(self.XY_net, XY, XY_ref).cpu().item()
        ce_X = _div(self.X_net, X, X_ref).cpu().item()
        ce_Y = _div(self.Y_net, Y, Y_ref).cpu().item()

        return ce_XY, ce_X, ce_Y

    def estimate(self, X=None, Y=None):
        r"""Return the mutual information estimate.
        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        dXY, dX, dY = self.forward(X, Y)
        return dXY - dX - dY

    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'X_net': self.X_net.state_dict(),
            'X_optimizer': self.X_optimizer.state_dict(),
            'Y_net': self.Y_net.state_dict(),
            'Y_optimizer': self.Y_optimizer.state_dict(),
            'X': self.X,
            'Y': self.Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ref_batch_factor': self.ref_batch_factor
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.X_net.load_state_dict(state_dict['X_net'])
        self.X_optimizer.load_state_dict(state_dict['X_optimizer'])
        self.Y_net.load_state_dict(state_dict['Y_net'])
        self.Y_optimizer.load_state_dict(state_dict['Y_optimizer'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        if 'lr' in state_dict:
            self.lr = state_dict['lr']
        if 'batch_size' in state_dict:
            self.batch_size = state_dict['batch_size']
        if 'ref_batch_factor' in state_dict:
            self.ref_batch_factor = state_dict['ref_batch_factor']
'''