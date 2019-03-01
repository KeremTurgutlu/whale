from fastai.vision import *
from fastai.basic_train import Recorder
from scipy.interpolate import UnivariateSpline

def smooth(alpha, x0, x1): return alpha*x0 + (1-alpha)*x1

def smooth_losses(losses, alpha=0.5):
    smoothed_losses = []
    x0 = losses[0]
    for i in np.arange(len(losses)-1):
        x0 = smooth(0.3, losses[i+1], x0)
        smoothed_losses.append(x0)
    return smoothed_losses
    
def plot_v2(self, skip_start=10, skip_end=5, alpha=0.5):
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
    lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
    losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
    # smoothing
    if alpha: losses, lrs = smooth_losses(losses, alpha), lrs[1:]
    _, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    mg = (np.gradient(np.array([x.item() for x in losses]))).argmin()
    print(f"Min numerical gradient: {lrs[mg]:.2E}")
    ax.plot(lrs[mg],losses[mg],markersize=10,marker='o',color='red')
    self.min_grad_lr = lrs[mg]

def smooth_by_spline(xs, ys, **kwargs):
    xs = np.arange(len(ys))
    spl = UnivariateSpline(xs, ys, **kwargs)
    ys = spl(xs)
    return ys

def plot_v3(self, skip_start=10, skip_end=5, **kwargs):
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
    lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
    losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
    losses = [x.item() for x in losses]
    losses = smooth_by_spline(lrs, losses, **kwargs)
    # smoothing
    _, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    mg = (np.gradient(np.array(losses))).argmin()
    print(f"Min numerical gradient: {lrs[mg]:.2E}")
    ax.plot(lrs[mg],losses[mg],markersize=10,marker='o',color='red')
    self.min_grad_lr = lrs[mg]

Recorder.plot_v2 = plot_v2
Recorder.plot_v3 = plot_v3














