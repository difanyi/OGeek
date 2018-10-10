import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Wrapper(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_input, num_output,
                 drop_keep_prob=0.6
                 ):
        super(MLP_Wrapper, self).__init__()

        self.num_layers = num_layers
        if type(hidden_dim) != list:
            self.hidden_dim = [hidden_dim for _ in range(num_layers)]
        self.num_input = num_input
        self.num_output = num_output

        self.drop_keep_prob = drop_keep_prob

        self.bulid_model()


    def bulid_model(self):

        self.in_fc = nn.Linear(self.num_input, self.hidden_dim[0])

        self.in_bn = nn.BatchNorm1d(self.hidden_dim[0])

        for i in range(self.num_layers -1):
            self.add_module('hidden_layer_{}'.format(i),
                            nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))

            self.add_module('bn_{}'.format(i), nn.BatchNorm1d(self.hidden_dim[i+1]))

        self.out_fc = nn.Linear(self.hidden_dim[-1], self.num_output)


    def forward(self, x):

        x = self.in_fc(x)

        x = self.in_bn(x)
        x = F.relu(x)

        for i in range(self.num_layers-1):
            x = self.__getattr__('hidden_layer_{}'.format(i))(x)
            x = self.__getattr__('bn_{}'.format(i))(x)

            x = F.relu(x)

        x = F.dropout(self.drop_keep_prob)

        out = self.out_fc(x)

        return out

    def compile_optimizer(self, opt, lr,  num_epochs, l2_reg = 0,
                          lr_decay=False, lr_decay_rate=0.9, lr_decay_min = None,
                          lr_decay_every = 1000,
                          ):

        self.num_epochs = num_epochs
        self.lr_decay = lr_decay
        self.lr_decay_rate=  lr_decay_rate
        self.lr_decay_min = lr_decay_min
        self.lr_decay_every = lr_decay_every

        if opt == 'adam':

            self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)

    def decay_lr(self):
        for group in self.opt.param_groups:
            old_lr = group['lr']
            new_lr = old_lr* self.lr_decay_rate

            if self.lr_decay_min: new_lr = max(self.lr_decay_min, new_lr)

            group['lr'] = new_lr

    def loss(self, logits, target):

        loss = F.cross_entropy(logits, target)
        return loss


    def fit(self, train_loader, val_loader, print_every=100,val_every= 1000):

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train()
        num_train_batches = len(train_loader)
        for epoch in range(self.num_epochs):
            for i ,(data, target) in enumerate(train_loader):
                step = num_train_batches * epoch + i

                if self.lr_decay and step >0 and step% self.lr_decay_every ==0:
                    self.decay_lr()

                logits= self.forward(data)
                loss = self.loss(logits, target)


                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                acc = (torch.argmax(logits, -1) == target).sum().float() / data.shape[0]
                if step %print_every ==0:

                    print('step:{0}, loss:{1}, acc:{2}'.format(step, loss, acc))

                if step >0 and step % val_every ==0:
                    loss, acc = self.evaluate()
                    print('validate at step {}, loss:{}, acc:{}'.format(step, loss, acc))
                    self.train()

    def evaluate(self):
        self.eval()
        losses, acc_ = [], []
        for data, target in self.val_loader:

            logits = self.forward(data)
            loss = self.loss(logits, target)

            acc = (torch.argmax(logits, -1) == target).sum().float() / data.shape[0]
            losses.append(loss)
            acc_.append(acc)


        return torch.mean(losses), torch.mean(acc_)

    def predict(self,x):
        logits = self.forward(x)

        pred = torch.argmax(logits, -1)

        return pred



