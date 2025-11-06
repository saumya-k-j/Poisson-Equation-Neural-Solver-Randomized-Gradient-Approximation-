import torch, torch.nn as nn, matplotlib.pyplot as plt, random
DEVICE = torch.device('cpu'); ALPHA=1e-2; T=10000; BATCH=1024; WIDTH=200; REWARD='x2'
torch.manual_seed(0); random.seed(0)
def sample_X0(b): return torch.rand(b, device=DEVICE)
def r_fn(x): return x if REWARD=='x' else x**2
def u_star(x): return 2.0*x if REWARD=='x' else (4/3)*x**2 + (2/3)*x
@torch.no_grad()
def step_P(x): z=torch.bernoulli(0.5*torch.ones_like(x)); return 0.5*(x+z)
@torch.no_grad()
def step_P2(x): x1=step_P(x); x2=step_P(x1); return x1,x2
class Perceptron(nn.Module):
    def __init__(self,width=1):
        super().__init__()
        self.net = nn.Linear(1,1) if width==1 else nn.Sequential(nn.Linear(1,width), nn.ReLU(), nn.Linear(width,1))
    def forward(self,x): return self.net(x.view(-1,1)).squeeze(-1)
@torch.no_grad()
def compute_metrics(net):
    grid = torch.linspace(0,1,401,device=DEVICE)
    u_hat = net(grid); u_gt=u_star(grid)
    c=(u_hat-u_gt).mean(); diff = u_hat-(u_gt+c); dx=grid[1]-grid[0]
    mse=(diff**2).mean().item(); mae=diff.abs().mean().item(); integral=(diff**2).sum().item()*dx.item()
    return mse, mae, integral, grid.cpu(), u_hat.cpu(), (u_gt+c).cpu()
def train_and_plot():
    net=Perceptron(WIDTH).to(DEVICE)
    for _ in range(1,T-1):
        X0=sample_X0(BATCH); X1,X2=step_P2(X0); Xm1,Xm2=step_P2(X0)
        u_X0,u_X1,u_X2=net(X0),net(X1),net(X2)
        r_X0,r_X1=r_fn(X0),r_fn(X1)
        g=(u_X0-2*u_X1+u_X2)-(r_X0-r_X1)
        u_Xm1,u_Xm2=net(Xm1),net(Xm2)
        h=u_X0-2*u_Xm1+u_Xm2
        loss=2*(g.detach()*h).mean()
        for p in net.parameters():
            if p.grad is not None: p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in net.parameters(): p -= ALPHA * p.grad
    net.eval()
    mse,mae,integral,grid,u_hat,u_shift=compute_metrics(net)
    print(f'MSE={mse:.6f}  MAE={mae:.6f}  ∫(error²)dx={integral:.6f}')
    plt.plot(grid,u_hat,label='Learned $u_θ(x)$'); plt.plot(grid,u_shift,label='True $u^*(x)+c$')
    plt.title(f'Algorithm 2 RGA  (Reward={REWARD}, width={WIDTH})'); plt.xlabel('x'); plt.ylabel('u(x)'); plt.legend(); plt.tight_layout(); plt.show()
if __name__=='__main__': train_and_plot()
