import numpy as np
import time





class RPGaussianPoolingNumpy:
    def __init__(self,
                 n_basis=8,
                 n_components=1, 
                 init_sigma=None,
                 use_normalization=False,
                 activation=None,
                 learnable_radius=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.learnable_radius = True
        self.init_sigma = init_sigma
        self.out_dim = n_components*(n_basis)**2
        # print('-----------init_sigma={}-------------'.format(init_sigma))
        


    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()

        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        if not self.init_sigma:
            init_sigma = np.sqrt(in_dim)
        else:
            init_sigma = self.init_sigma



        # self.E = np.random.random([in_dim, int(self.n_basis*np.sqrt(self.n_components))])
        # self.F = np.random.random([in_dim, int(self.n_basis*np.sqrt(self.n_components))])

        for n in range(self.n_components):
            
            E0 = np.random.random([in_dim, self.n_basis])
            # sigma_e = 0.1
            # self.E_list.append( np.sqrt(in_dim) / (1e-6 + sigma_e) * E0  )
            self.E_list.append( E0  )

            F0 = np.random.random([in_dim, self.n_basis])
            # sigma_f = 0.1
            # self.F_list.append( np.sqrt(in_dim) / (1e-6 + sigma_f) * F0  )
            self.F_list.append( F0  )



    def call(self, X):
        self.build(X.shape)

        ### here X is the entire mat with [batch, T, D]
        n_frames = X.shape[1]
        in_dim = float(X.shape[-1])
        z_list = []
        # z1 = np.matmul(X, self.E )
        # z2 = np.matmul(X, self.F )


        z_list = list(map( lambda x: np.reshape(np.einsum( 'ijm, ijn->ijmn', 
                                                 np.matmul(X, x[0]),
                                                 np.matmul(X, x[1])),
                                                [X.shape[0], n_frames, -1]), 
                            zip(self.E_list, self.F_list)
                        )
                )


        # for ii in range(self.n_components):

            # # outer product
            # z1 = np.expand_dims(z1, axis=-1)
            # z2 = np.expand_dims(z2, axis=-2)
            # z12 = np.matmul(z1, z2)
            # z12 = np.reshape(z12, [-1, n_frames, (self.n_basis)**2])
        # z12 = np.einsum('ijm, ijn->ijmn', z1, z2)
        # z12 = np.reshape(z12, [X.shape[0], n_frames, -1])
            # print(z12.shape)
            # z_list.append(z12)

        z = np.concatenate(z_list, axis=-1)
        return z



    # def call(self, X):
    #     self.build(X.shape)

    #     ### here X is the entire mat with [batch, T, D]
    #     n_frames = X.shape[1]
    #     in_dim = float(X.shape[-1])
    #     z_list = []




    #     for ii in range(self.n_components):

    #         z1 = np.matmul(X, self.E_list[ii] )
    #         z2 = np.matmul(X, self.F_list[ii] )

    #         # outer product
    #         z1 = np.expand_dims(z1, axis=-1)
    #         z2 = np.expand_dims(z2, axis=-2)
    #         z12 = np.matmul(z1, z2)
    #         z12 = np.reshape(z12, [-1, n_frames, (self.n_basis)**2])

    #         z_list.append(z12)

    #     z = np.concatenate(z_list, axis=-1)
        
    #     return z



class MLBNumpy:
    def __init__(self,
                 n_basis=8,
                 n_components=1, 
                 init_sigma=None,
                 use_normalization=False,
                 activation=None,
                 learnable_radius=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.learnable_radius = True
        self.init_sigma = init_sigma
        self.out_dim = n_components*(n_basis)**2
        # print('-----------init_sigma={}-------------'.format(init_sigma))
        


    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]


        ## define the two matrix with orthogonal columns
        
        
            
        self.E0 = np.random.random([in_dim, self.n_basis])

        self.F0 = np.random.random([in_dim, self.n_basis])



    def call(self, X):
        self.build(X.shape)

        ### here X is the entire mat with [batch, T, D]
        n_frames = X.shape[1]
        in_dim = float(X.shape[-1])
        xx = np.matmul(X, self.E0)
        yy = np.matmul(X, self.F0)

        z = xx*yy
        
        return z



factor = [1, 2, 4]
in_dim = 64
X = np.random.random([8,128,64]) # batch=8, dim=32

timefun = time.process_time

for ff in factor:

    time_rp_list = []
    time_mlb_list = []

    for ii in range(100):

        rp_pooling = RPGaussianPoolingNumpy(n_basis = in_dim//2, n_components=ff)
        rp_mlb = MLBNumpy(n_basis = in_dim**2//4*ff)
        
        t0 = timefun()
        z = rp_pooling.call(X)
        t1 = timefun()
        z2 = rp_mlb.call(X)
        t2 = timefun()

        time_rp_list.append(t1-t0)
        time_mlb_list.append(t2-t1)
    print(z.shape)
    print('--n_components={}, [RPGaussian] mean cpu time={}'.format(ff, np.mean(time_rp_list)))
    print('--n_components={}, [MLB] mean cpu time={}'.format(ff, np.mean(time_mlb_list)))


