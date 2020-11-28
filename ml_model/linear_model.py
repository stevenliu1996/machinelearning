#! /usr/bin/
# by yanbin Liu
import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept = False):
        self.intercept_ = 0 if fit_intercept else None
        self.coef_ = None

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        if self.intercept_ is not None:
            x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
            w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
            self.coef_ = w[1:]

    def predict(self, x):
        if self.coef_ is None:
            raise ValueError('model not fitted yet')
        return np.dot(x, self.coef_) + [self.coef_, 0][self.coef_ is None]
        
    def r2_score(self, x, y):
        if self.coef_ is None:
            raise ValueError('model not fitted yet')
        res = y - self.predict(x)
        return 1 - (res**2).sum() / ((y-y.mean())**2).sum()

    def fit_GD(self, X, Y, alpha = 0.001, verbose=False, max_iterator=1000, tol=0.000001):
        if self.intercept_ is not None:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        assert X.shape[0] == Y.shape[0]
        m = X.shape[1]
        w1 = np.ones(m)
        bad_count, previous_diff = 0, 0 
        success = False
        for i in range(max_iterator):
            w2 = w1 - alpha * X.T.dot(X.dot(w1)-Y)
            if ((w1-w2)**2).sum() < tol:
                success=True
                break
            if verbose:
                print('loss:', ((w1-w2)**2).sum())
            
            bad_count, previous_diff = bad_count +1 if ((w1-w2)**2).sum() - previous_diff > 0 else 0, ((w1-w2)**2).sum()
            if bad_count > 20:
                break
            w1 = w2
        if success:
            model.coef_ = w1[1:] if self.intercept_ is not None else w1
            model.intercept_ = w1[0] if self.intercept_ is not None else None
            return 'succeed'
        else:
            print('Not Converge')
            return 'fail'


    def fit_GDB(self, X, Y, alpha = 0.001, verbose=False, max_iterator=1000, tol=0.000001, batch=100):
        assert X.shape[0] == Y.shape[0]
        if self.intercept_ is not None:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        m = X.shape[1]
        w1 = np.ones(m)  
        bad_count, previous_diff = 0, 0 
        success = False
        for i in range(max_iterator):
            num_of_batch = X.shape[0] // batch +1
            w2 = w1.copy()
            for j in range(num_of_batch):
                idx = np.arange(j * batch, np.minimum((j+1) * batch, X.shape[0]))
                w2 = w2 - alpha * X[idx, :].T.dot(X[idx, :].dot(w1)-Y[idx])  
                if (j % np.maximum(num_of_batch // 5, 1) == 0) & verbose:
                    print('{:.0f}%'.format(j/num_of_batch*100),end=' ')
            if verbose:
                print(((w1-w2)**2).sum())
            if ((w1-w2)**2).sum() < tol:
                success=True
                break
            bad_count, previous_diff = bad_count +1 if ((w1-w2)**2).sum() - previous_diff > 0 else 0, ((w1-w2)**2).sum()
            if bad_count > 20:
                break
            w1 = w2
        if success:
            model.coef_ = w1[1:] if self.intercept_ is not None else w1
            model.intercept_ = w1[0] if self.intercept_ is not None else None
            return 'succeed'
        else:
            print('Not Converge')
            return 'fail'

class Lasso():
    def __init__(self, fit_intercept = False, alpha=1):
        self.intercept_ = 0 if fit_intercept else None
        self.coef_ = None
        self.alpha = alpha

    def fit(self, X, Y, learning_rate=0.0001, verbose=False, max_iterator=1000, tol=0.000001):
        if self.intercept_ is not None:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        assert X.shape[0] == Y.shape[0]
        m = X.shape[1]
        w1 = np.ones(m)
        bad_count, previous_diff = 0, 0 
        success = False
        for i in range(max_iterator):
            w2 = w1 - learning_rate * (X.T.dot(X.dot(w1)-Y) + self.alpha * np.sign(w1))
            if ((w1-w2)**2).sum() < tol:
                success=True
                break
            if verbose:
                print('loss:', ((w1-w2)**2).sum())
            
            bad_count, previous_diff = bad_count +1 if ((w1-w2)**2).sum() - previous_diff > 0 else 0, ((w1-w2)**2).sum()
            if bad_count > 20:
                break
            w1 = w2
        if success:
            model.coef_ = w1[1:] if self.intercept_ is not None else w1
            model.intercept_ = w1[0] if self.intercept_ is not None else None
            return 'succeed'
        else:
            print('Not Converge')
            return 'fail'

if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    # n = 475200
    # m = 1000
    # X = np.random.randn(n,m)
    # Y = X.dot(np.linspace(1, 2, m))+np.random.randn(n)+2

    # linear algebra 46s
    # model = LinearRegression(fit_intercept=True)
    # model.fit(X, Y)
    # sns.scatterplot(x=range(m), y=model.coef_, color = 'red')
    # sns.lineplot(x=range(m), y=np.linspace(1,2, m))
    # plt.show()

    # gradient descent 23.6s
    # model = LinearRegression(fit_intercept=True)
    # model.fit_GD(X, Y, alpha=0.000001, verbose=True)
    # print(model.intercept_)
    # sns.scatterplot(x=range(m), y=model.coef_, color = 'red')
    # sns.lineplot(x=range(m), y=np.linspace(1,2, m))
    # plt.show()

    # gradient descent with batches 31.6s
    # model = LinearRegression(fit_intercept=True)
    # model.fit_GDB(X, Y, alpha=0.000001, verbose=True)
    # print(model.intercept_)
    # sns.scatterplot(x=range(m), y=model.coef_, color = 'red')
    # sns.lineplot(x=range(m), y=np.linspace(1,2, m))
    # plt.show()

    # gradient descent for lasso 25s
    n = 470000
    m = 1000
    X = np.random.randn(n,m)
    good_feature = 300
    Y = X[:, :good_feature].dot(np.linspace(1, 2, good_feature))+np.random.randn(n) + 3
    model = Lasso(fit_intercept=True, alpha=1)
    model.fit(X, Y, learning_rate=0.000001, verbose=True)
    print(model.intercept_)
    sns.scatterplot(x=range(m), y=model.coef_, color = 'red')
    sns.lineplot(x=range(good_feature), y=np.linspace(1,2, good_feature))
    plt.show()

