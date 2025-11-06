import time, random, torch, torch.nn as nn, plotly.graph_objects as go, streamlit as st
DEVICE=torch.device('cpu')
def r_fn(x, reward): return x if reward=='x' else x**2
def u_star(x, reward): return 2.0*x if reward=='x' else (4/3)*x**2 + (2/3)*x
@torch.no_grad()
def step_P(x): z=torch.bernoulli(0.5*torch.ones_like(x)); return 0.5*(x+z)
@torch.no_grad()
def step_P2(x): x1=step_P(x); x2=step_P(x1); return x1,x2
class Perceptron(nn.Module):
    def __init__(self,width=200): super().__init__(); self.net=nn.Sequential(nn.Linear(1,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,x): return self.net(x.view(-1,1)).squeeze(-1)
@torch.no_grad()
def compute_metrics(net, reward):
    grid=torch.linspace(0,1,401,device=DEVICE); u_hat=net(grid); u_gt=u_star(grid,reward); c=(u_hat-u_gt).mean()
    diff=u_hat-(u_gt+c); dx=grid[1]-grid[0]; mse=(diff**2).mean().item(); mae=diff.abs().mean().item(); integral=(diff**2).sum().item()*dx.item()
    return dict(mse=mse,mae=mae,integral=integral,grid=grid.cpu().numpy(),u_hat=u_hat.cpu().numpy(),u_shift=(u_gt+c).cpu().numpy())
def train_once(T,ALPHA,BATCH,WIDTH,REWARD,seed,progress_cb=None,refresh_every=200):
    torch.manual_seed(seed); random.seed(seed); net=Perceptron(WIDTH).to(DEVICE)
    for t in range(1,T+1):
        X0=torch.rand(BATCH,device=DEVICE); X1,X2=step_P2(X0); Xm1,Xm2=step_P2(X0)
        u_X0,u_X1,u_X2=net(X0),net(X1),net(X2); r_X0,r_X1=r_fn(X0,REWARD),r_fn(X1,REWARD)
        g=(u_X0-2*u_X1+u_X2)-(r_X0-r_X1); u_Xm1,u_Xm2=net(Xm1),net(Xm2); h=u_X0-2*u_Xm1+u_Xm2; loss=2*(g.detach()*h).mean()
        for p in net.parameters():
            if p.grad is not None: p.grad.zero_()
        loss.backward(); 
        with torch.no_grad():
            for p in net.parameters(): p -= ALPHA * p.grad
        if progress_cb and (t % refresh_every==0 or t==T): progress_cb(t, compute_metrics(net,REWARD))
    return net
st.set_page_config(page_title='Neural Poisson Solver (RGA)', layout='centered')
st.title('Neural Poisson Equation Solver — RGA (1D)')
with st.sidebar:
    reward=st.radio('Reward r(x)', ['x2','x'], index=0); T=st.slider('Iterations (T)',200,20000,5000,step=200)
    alpha=st.select_slider('Learning rate (α)', options=[1e-3,2e-3,5e-3,1e-2,2e-2,5e-2], value=1e-2)
    batch=st.select_slider('Batch size', options=[128,256,512,1024,2048], value=1024)
    width=st.select_slider('Hidden width', options=[50,100,200,400], value=200)
    seed=st.number_input('Seed', min_value=0, value=0, step=1); live=st.checkbox('Live updates', value=True)
    refresh_every=st.select_slider('Update every N steps', options=[50,100,200,500,1000], value=200)
start=st.button('Train'); plot_area=st.empty(); mse_box, mae_box, int_box = st.columns(3)
def render_plot(m): 
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=m['grid'], y=m['u_hat'], mode='lines', name='Learned uθ(x)'))
    fig.add_trace(go.Scatter(x=m['grid'], y=m['u_shift'], mode='lines', name='True u*(x)+c'))
    fig.update_layout(width=800,height=420,title=f'RGA – Reward={reward}, width={width}',xaxis_title='x',yaxis_title='u(x)',template='plotly_white',legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
    plot_area.plotly_chart(fig, use_container_width=True)
def render_metrics(m): mse_box.metric('MSE', f"{m['mse']:.6f}"); mae_box.metric('MAE', f"{m['mae']:.6f}"); int_box.metric('∫(error²)dx', f"{m['integral']:.6f}")
progress=st.progress(0, text='Ready')
if start:
    import pandas as pd, time
    def cb(step,metrics): 
        if live: render_plot(metrics); render_metrics(metrics)
        progress.progress(min(step/T,1.0), text=f'Training... {step}/{T}')
    net=train_once(T,alpha,batch,width,reward,seed,progress_cb=cb,refresh_every=refresh_every)
    final=compute_metrics(net,reward); render_plot(final); render_metrics(final); progress.progress(1.0, text='Done')
    df=pd.DataFrame([{'T':T,'alpha':alpha,'batch':batch,'width':width,'reward':reward,'mse':final['mse'],'mae':final['mae'],'integral':final['integral']}])
    st.download_button('Download metrics CSV', df.to_csv(index=False).encode('utf-8'), file_name='rga_metrics_single_run.csv', mime='text/csv')
st.caption('Tip: α≈0.05 often converges fast; larger batch → smoother updates.')
