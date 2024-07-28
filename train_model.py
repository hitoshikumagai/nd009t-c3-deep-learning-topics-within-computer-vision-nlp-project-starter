#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import json, os, argparse
import smdebug.pytorch as smd

def test(model, test_loader, criterion, device, hook, mode='test'):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.EVAL)
    hook.register_loss(criterion)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    if mode == 'valid':
        print(
            "\n Valid Loss: valid_loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
            )
        )        
    else:
        print(
            "\n Test Loss: testing_loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
            )
        )


def train(model, train_loader, valid_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook.set_mode(smd.modes.TRAIN)
    hook.register_loss(criterion)
    for e in range(args.epochs):
        model.train()
        running_loss = 0
        correct = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss += loss

            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
             Accuracy {100*(correct/len(train_loader.dataset))}%")
            
        test(model, valid_loader, criterion, device, hook, 'valid')
        
    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model


def create_data_loaders(args):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #preprocess reference: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    training_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))
        ])

    testing_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))
        ])

    channels = args.data

    train_dir = channels + '/train/'
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dir = channels + '/valid/'
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=testing_transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dir = channels + '/test/'
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=testing_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

def main(args):
    
    train_loader, valid_loader, test_loader = create_data_loaders(args)

    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = train(model, train_loader, valid_loader, criterion, optimizer, device, hook)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device, hook)

    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':

    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default:512)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default:1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default:10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate(default:1.0)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="LR", help="momentum(default:0.9)"
    )
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args)
    main(args)