#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

import sagemaker
import boto3
from PIL import Image
import io

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target, reduction="sum")
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)

    for e in range(args.epochs):
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


def create_data_loaders(batch_size, prefix):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    bucket = "sagemaker-studio-146700155215-5c7uj0emdok"
    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket)
    s3_client = boto3.client('s3')

    object_key = [obj.key for obj in s3_bucket.objects.all()]
    data_keys = [obj for obj in object_key if f'{prefix}/' in obj]

    X = []
    for idx, data_key in enumerate(data_keys):
        response = s3_client.get_object(Bucket=bucket, Key=data_key)
        image_data = response['Body'].read()

        size = 256, 256
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            im_resized = image.resize(size)
            im_resized = np.array(im_resized).astype(np.float32).transpose(2, 1, 0)
        except OSError as e:
            print(f"Error opening image: {e}")

        X.append(im_resized)
    X = torch.tensor(X)
    y_df = pd.read_csv(f's3://{bucket}/nd009t-c2/metadata/{prefix}_metadata.csv', delimiter='\t')
    y = y_df['label'].values
    y = torch.LongTensor(y)

    Dataset = torch.utils.data.TensorDataset(X, y)
    Loader = torch.utils.data.DataLoader(dataset=Dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2)
    return Loader


def main(args):
    train_loader = create_data_loaders(args.batch_size, 'valid')
    test_loader = create_data_loaders(args.test_batch_size, 'test')

    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()

    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model = train(model, train_loader, criterion, optimizer)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)

    '''
    TODO: Save the trained model
    '''
    #torch.save(model, path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default:256)",
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
    args = parser.parse_args()
    main(args)