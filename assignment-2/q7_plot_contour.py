import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression


np.random.seed(42)

X = np.array([i*np.pi/180 for i in range(60,300,4)])
y = 4*(X) + 7 + np.random.normal(0,3,len(X))

LR = LinearRegression(fit_intercept=True)

# varying lr_type and plotting
for lr_type in ['constant', 'inverse']:
    LR.fit_autograd(pd.DataFrame(np.array([list(X)]).T), pd.Series(y), 60, n_iter=10, lr=0.02, lr_type=lr_type)

    anim = LR.plot_line_fit(pd.DataFrame(np.array([list(X)]).T), pd.Series(y))
    anim.save('images/line_plot_'+lr_type+'.gif', dpi=80, writer='imagemagick')
    plt.show()

    anim1 = LR.plot_contour(pd.DataFrame(np.array([list(X)]).T), pd.Series(y))
    anim1.save('images/contour_'+lr_type+'.gif', dpi=80, writer='imagemagick')
    plt.show()

    anim2 = LR.plot_surface(pd.DataFrame(np.array([list(X)]).T), pd.Series(y))
    anim2.save('images/surface_'+lr_type+'.gif', dpi=80, writer='imagemagick')
    plt.show()