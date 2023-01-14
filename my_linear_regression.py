import numpy as np

# x = np.arange (-10, 10)
# y = 1 + 2 * x
# y2 = 1 + 1000 * x
# plt.figure(figsize = (2,5))
# plt.plot(x,y, "r")
# plt.plot(x,y2, "b")
# plt.grid()

def h( x, theta ):
    return np.array([[i] for i in np.dot(x,theta)])

def bias_column( x ):
    theta = np.array([1,2])
    new_col = np.ones((100,1))
    x = x.reshape(100,1)
    x = np.append(new_col, x, axis = 1)
    return x

def mean_squared_error( y_pred, y_label ):
  return np.sum((y_pred - y_label )**2)/len(y_label)

class LeastSquaresRegression():
    def __init__( self ):
        self.theta_ = None  
        
    def fit( self, X, y ):
        #θ = (XT·X)-1·XT·y
        part_1 = np.dot(X.T, X)
        part_2 = np.dot(X.T, y)
        my_inv = np.linalg.inv(part_1)
        self.theta_ = np.dot(my_inv, part_2)

    
        
    def predict( self, X ):
        return h(X, self.theta_)
        
# X = 4 * np.random.rand(100, 1)
# y = 10 + 2 * X + np.random.randn(100, 1)
# plt.scatter(X, y)
# model = LeastSquaresRegression()
# model.fit(X,y)
# model.theta_
# X = np.append(np.ones((100,1)), X, axis=1)

class GradientDescentOptimizer():

    def __init__( self, f, fprime, start, learning_rate = 0.1 ):
        self.f_      = f                       
        self.fprime_ = fprime                 
        self.current_ = start                  
        self.learning_rate_ = learning_rate    
        self.history_ = start
    
    def step( self ):
        self.current_ = self.current_ - self.learning_rate_ * fprime(self.current_)
        self.history_ = np.append(self.history_, self.current_,axis = 1)
        
    def optimize( self, iterations = 100 ):
        iters = 0
        while iters < iterations:
            self.step()
            iters += 1

    def print_result( self ):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))

def f( x ):
    a = np.array([2, 6])
    return 3 + np.dot((x - a).T,(x - a))

def fprime( x ) :
    a = np.array([2,6])
    return 2 * (x - a)

grad = GradientDescentOptimizer(f, fprime, np.random.normal(size = (2,1)),0.1)
grad.optimize(10)
grad.print_result()