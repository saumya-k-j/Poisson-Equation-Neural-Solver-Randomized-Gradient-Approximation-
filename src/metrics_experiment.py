import os, random, pandas as pd, torch
from poisson_rga import Perceptron, step_P2, r_fn, compute_metrics, DEVICE, WIDTH
def train_once(T:int, ALPHA:float, BATCH:int, seed:int=0):
    torch.manual_seed(seed); random.seed(seed); net=Perceptron(WIDTH).to(DEVICE); net.train()
    for _ in range(T):
        import torch as th
        X0=th.rand(BATCH, device=DEVICE); X1,X2=step_P2(X0); Xm1,Xm2=step_P2(X0)
        u_X0,u_X1,u_X2=net(X0),net(X1),net(X2); r_X0,r_X1=r_fn(X0),r_fn(X1)
        g=(u_X0-2*u_X1+u_X2)-(r_X0-r_X1); u_Xm1,u_Xm2=net(Xm1),net(Xm2); h=u_X0-2*u_Xm1+u_Xm2
        loss=(2*(g.detach()*h)).mean()
        for p in net.parameters():
            if p.grad is not None: p.grad.zero_()
        loss.backward()
        with th.no_grad():
            for p in net.parameters(): p -= ALPHA * p.grad
    net.eval(); mse,mae,integral,*_=compute_metrics(net); return {'mse':mse,'mae':mae,'integral':integral}
if __name__=='__main__':
    T_LIST=[500,2000,5000,10000]; ALPHA_LIST=[1e-3,1e-2,5e-2]; BATCH_LIST=[256,1024]
    rows=[]; run=0
    for T in T_LIST:
        for A in ALPHA_LIST:
            for B in BATCH_LIST:
                run+=1; m=train_once(T,A,B)
                rows.append({'run':run,'T':T,'alpha':A,'batch':B,**m})
                print(f'Run {run}: T={T}, α={A}, batch={B} -> MSE={m["mse"]:.6f}, MAE={m["mae"]:.6f}, ∫={m["integral"]:.6f}')
    outdir=os.path.join(os.path.dirname(__file__),'..','data'); os.makedirs(outdir,exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir,'metrics_table.csv'),index=False); print('Saved data/metrics_table.csv')
