from pathlib import Path
import torch
import torch.nn as nn

Features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

#load at import time
_ckpt=torch.load(Path(__file__).with_name("housing_linreg.pt"), map_location="cpu")
_model=nn.Linear(len(Features),1)
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()
_xmean,_xstd=_ckpt["xmean"],_ckpt["xstd"]

def predict(features: dict)->float:
    x=torch.tensor([[features[name] for name in Features]],dtype=torch.float32)
    x=(x-_xmean)/_xstd
    with torch.no_grad():
        pred=_model(x).item()
    return max(pred,0.0)*100_000
