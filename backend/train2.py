import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.datasets import fetch_california_housing

device="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(78)

X,y=fetch_california_housing(return_X_y=True)

X_all=torch.tensor(X,dtype=torch.float32,device="cuda")
Y_all=torch.tensor(y,dtype=torch.float32,device="cuda")

#splitting 80/20 test train
perm=torch.randperm(len(X_all),device=device)
n_test=int(0.2*len(X_all))
te,tr=perm[:n_test],perm[n_test:]
X_tr,y_tr,X_te,y_te=X_all[tr],Y_all[tr],X_all[te],Y_all[te]

#normalizing
x_mean=X_tr.mean(dim=0,keepdim=True)
x_std=X_tr.std(dim=0,keepdim=True)
X_tr_n=(X_tr-x_mean)/x_std
X_te_n=(X_te-x_mean)/x_std

        #creating our linear regression class
class Linearreg(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features)

    def forward(self,X):
        return self.linear(X).squeeze(1)

#defining our loss function in this case huber loss
in_features=8
out_features=1

model=Linearreg(in_features,out_features).to(device)

loss=nn.HuberLoss(delta=1.0,reduction="mean")
optimizer=optim.SGD(model.parameters(),lr=0.1)

#training loop
epochs=1000
for epoch in range(epochs):
    model.train()
    output=model(X_tr_n)

           #calculating our loss for each ep:och 
    losscalc=loss(output,y_tr)

    #zeroing out gradients
    optimizer.zero_grad()

    #backward pass
    losscalc.backward()
    optimizer.step()

    if epoch%100==0:
        print(f'Epoch{epoch},Loss: {losscalc}')

#evaluvating the model
model.eval()
with torch.no_grad():
    pred=model(X_te_n)
    err=y_te-pred
    mse=(err**2).mean().item()
    mae=err.abs().mean().item()
    r2 = 1 - (err ** 2).sum().item() / ((y_te - y_te.mean()) ** 2).sum().item()
    baseline_mse = ((y_te - y_tr.mean()) ** 2).mean().item()
    idx = torch.arange(0, len(y_te), max(1, len(y_te) // 300))

print(f"Test MSE{mse:.3f}, MAE{mae:.3f},R2{r2:.3f},baseline MSE{baseline_mse:.3f}")

#dumping json with evaluvation metrics
with open("metrics.json","w")as f:
    json.dump({"n_test":len(y_te),"mse":mse,"rmse":mse**0.5,"mae":mae,
           "r2":r2,"baseline_mse":baseline_mse,
           "points":{"actual":y_te[idx].tolist(),"predicted":pred[idx].tolist()}
    },f)
#saving the model 
torch.save({
    "state_dict": model.linear.state_dict(),
    "xmean": x_mean.cpu(),
    "xstd": x_std.cpu(),
}, "housing_linreg.pt")
