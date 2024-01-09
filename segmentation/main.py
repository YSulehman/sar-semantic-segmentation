import os
import torch
import argparse
import torch.nn as nn
from src import dataset, trainer, evaluate
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


# import torchaudio


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
    val_split = args.val_split

    if args.save_dir is None:
        # we will save the trained model in same location as city data
        save_dir = pth
    else:
        save_dir = args.save_dir

    print('creating model...')
    if encoder_weights:
        model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels,
                         classes=out_channels)
    else:
        #aux_params = dict(dropout=0.2, classes=out_channels)
        model = smp.Unet(encoder_name=encoder, in_channels=in_channels, classes=out_channels)

    # update this to split into train and validation
    data = dataset.SegDataset(pth, vert_split=False)
    #split into train and val
    train_size = int((1-val_split) * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch, shuffle=False)

    # allow for choice of loss and optimiser?
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # train model
    t = trainer.Trainer(model, train_loader, val_loader, loss_function, optimiser, epochs, device, save_dir)
    print('beginning training....')
    t.train_model()
    print('training complete!')

    # evaluate model
    if model_eval:
        # pass pth in general
        model_pth = '/ste/rnd/User/yusuf/city_data/Berlin_Summer/trained_model.pth'
        test_data = dataset.SegDataset(pth,
                                       vert_split=False, train=False)
        test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)
        # can normally pass t.file as 2nd argument
        e = evaluate.Evaluate(model, model_pth, test_loader, test_data.hex_rgb, device, save_dir)
        print('beginning evaluation....')
        e.perform_evaluation()
        print('evaluation complete!')


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

    # parser.add_argument('-opt', '--optimiser', type=str, default='adams', help='optimiser for training')

    parser.add_argument('-bch', '--batch_size', type=int, default=11,
                        help='batch size for train loader')

    parser.add_argument('-eps', '--epochs', type=int, default=100,
                        help='number of training epochs')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for training')

    parser.add_argument('-vs', '--val_split', type=float, default=0.2,
                        help='validation split')

    parser.add_argument('-dev', '--device', type=str, default='cuda',
                        help='set the device for training')

    parser.add_argument('-sd', '--save_dir', type=str, default=None, help='directory in which to save trained model')

    # evaluation argument
    parser.add_argument('-ev', '--eval', action="store_true",
                        help='specify if evaluation metrics are required')

    args = parser.parse_args()

    main(args)

    # print(torch.backends.cudnn.enabled)
    # if torch.cuda.is_available():
    #     print(f'number of gpus available is: {torch.cuda.device_count()}')
    # else:
    #     print('no gpus')
    # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
