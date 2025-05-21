import torch
model = torch.nn.Linear(10, 2)  # dummy model
torch.save(model.state_dict(), "shahid_model.pth")
