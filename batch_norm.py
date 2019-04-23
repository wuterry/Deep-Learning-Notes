from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
from utils import *


class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.eps = 1.0e-10
        self.momentum = 0.9
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)

        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())

        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X,  is_training=True):
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)

        # inference only
        if not is_training:
            X_hat = (X - self.moving_mean) / nd.sqrt(self.moving_var + self.eps)

        # training
        else:
            assert len(X.shape) in (2, 4)

            # case of full connected layer
            if len(X.shape) == 2:
                mean = X.mean(axis=0)
                var = ((X - mean) ** 2).mean(axis=0)

            # case of convolution layer
            if len(X.shape) == 4:
                mean = X.mean(axis=(0, 2, 3), keepdims=True)
                var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            X_hat = (X - mean) / nd.sqrt(var + self.eps)

            # update moving_mean and moving_var
            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * var

        Y = self.gamma.data() * X_hat + self.beta.data()
        return Y

def Lenet():
    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            BatchNorm(6, num_dims=4),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5),
            BatchNorm(16, num_dims=4),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120),
            BatchNorm(120, num_dims=2),
            nn.Activation('sigmoid'),
            nn.Dense(84),
            BatchNorm(84, num_dims=2),
            nn.Activation('sigmoid'),
            nn.Dense(10))
    return net


def main():

    net = Lenet()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size

        test_acc = evaluate_accuracy(test_iter, net, ctx=ctx)
        print('epoch=%d\tloss=%.4f\ttrain-acc=%.3f\ttest-acc=%.3f\ttime=%.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))


if __name__ == '__main__':
    lr = 1.0
    num_epochs = 5
    batch_size = 32
    ctx = mx.gpu(0)
    main()