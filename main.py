import os
import torch
import argparse
import torch.nn as nn
from src import dataset, trainer, evaluate
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


def main(args):
    pth = args.input_file
    encoder = args.backbone
    encoder_weights = args.pre_trained
    in_channels = args.input_channels
    out_channels = args.output_channels
    epochs = args.epochs
    batch = args.batch_size
    lr = args.learning_rate
    device = torch.device(args.device)
    model_eval = args.eval
    if args.save_dir is None:
        #we will save the trained model in same location as city data
        save_dir = pth
    else:
        save_dir = args.save_dir

    if encoder_weights:
        model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels,
                         classes=out_channels)
    else:
        model = smp.Unet(encoder_name=encoder, in_channels=in_channels, classes=out_channels)

    #pass pth in general
    data = dataset.SegDataset('../../../ste/rnd/User/yusuf/city_data/Berlin_Summer', vert_split=False)
    train_loader = DataLoader(data, batch_size=batch, shuffle=True)

    # re-write this block
    # if args.loss == "cross_entropy":
    #     loss_function = nn.CrossEntropyLoss()
    #
    # if args.optimiser == "adams":
    #     optimiser = torch.optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    #train model
    t = trainer.Trainer(model, train_loader, loss_function, optimiser, epochs, device, save_dir)
    t.train_model()

    #evaluate model
    if model_eval:
        #pass pth in general
        test_data = dataset.SegDataset('../../../ste/rnd/User/yusuf/city_data/Berlin_Summer',
                                       vert_split=False, train=False)
        test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)
        e = evaluate.Evaluate(model, t.file, test_data, device, save_dir)
        e.perform_evaluation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for data loading script
    parser.add_argument('-f', '--input_file', type=str, required=True, help='path to tif file')

    parser.add_argument('-b', '--backbone', type=str, default='resnet18', help='model encoder')

    parser.add_argument('-pt', '--pre_trained', action="store_const", const='imagenet', default=False,
                        help='used pre-trained imagenet weights')

    parser.add_argument('-ic', '--input_channels', type=int, required=True, help='number of input channels')

    parser.add_argument('-oc', '--output_channels', type=int, required=True, help='number of classes')

    # arguments for training
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        help='loss function, default is cross entropy loss')

    #parser.add_argument('-opt', '--optimiser', type=str, default='adams', help='optimiser for training')

    parser.add_argument('-bch', '--batch_size', type=int, default=10,
                        help='batch size for train loader')

    parser.add_argument('-eps', '--epochs', type=int, default=100,
                        help='number of training epochs')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for training')

    parser.add_argument('-dev', '--device', type=str, default='cuda',
                        help='set the device for training')

    parser.add_argument('-sd', '--save_dir', type=str, default=None, help='directory in which to save trained model')

    #evaluation argument
    parser.add_argument('-ev', '--eval', action="store_true",
                        help='specify if evaluation metrics are required')

    args = parser.parse_args()

    main(args)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # print(torch.cuda.is_available())
    #
    # num_gpus = torch.cuda.device_count()
    #
    # print(f'number of available gpus is: {num_gpus}')

