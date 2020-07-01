import torch
from torch_geometric.data import DataLoader
from neuralnet_pytorch.metrics import chamfer_loss
from Compeletion3D import Completion3D
from Models import SA_Net




def train():
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        step += 1
        if step % 50 == 0:
            print(step)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.reshape(-1,2048,3), data.x.reshape(-1,2048,3))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)



if __name__ == '__main__':

    path = '../../data/shapenet_2048'
    dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    print(dataset[0])
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SA_Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    print('Training started:')
    criterion = chamfer_loss()
    for epoch in range(1, 401):
        loss = train()
        print('Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        if epoch % 10 ==0:
            torch.save(model.state_dict(),'./pointAECh'+'{}'.format(epoch)+'.pt')