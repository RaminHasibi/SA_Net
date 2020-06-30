import torch
from torch_geometric.data import DataLoader
from chamfer_distance import ChamferDistance
from Completion3D import Completion3D





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
        dist1, dist2 = criterion(out.reshape(-1,2048,3), data.x.reshape(-1,2048,3))
        loss = (torch.mean(dist1)) + (torch.mean(dist2)) 
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)



if __name__ == '__main__':

    path = '../../data/shapenet_2048'
    dataset = ShapeNet_2048(path, split='trainval', categories='Chair')
    print(dataset[0])
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    print('Training started:')
    criterion = ChamferDistance()
    for epoch in range(1, 401):
        loss = train()
        print('Epoch {:03d}, Loss: {:.4f}'.format(
            epoch, loss))
        if epoch % 10 ==0:
            torch.save(model.state_dict(),'./pointAECh'+'{}'.format(epoch)+'.pt')