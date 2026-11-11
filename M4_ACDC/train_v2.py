import os
import time
import datetime
import torch
from acdc_dataset import ACDCDataset
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from src.unet2 import TinyUnet
import src.unet2 as m

def create_model(num_classes):
    model = TinyUnet(in_ch=1, num_classes=num_classes)
    return model

def dice_loss(pred,target,smooth=1.0):
    pred =torch.sigmoid(pred)
    inter =(pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3))+target.sum(dim=(2,3))
    return 1-((2*inter+smooth)/(union+smooth)).mean()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    train_dataset = ACDCDataset(r"D:\M4_ACDC\training",train=True)

    val_dataset = ACDCDataset(r"D:\M4_ACDC\training",train=False)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model(num_classes=1)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)  
    start_time =time.time()
    best_dice = 0.
    for epoch in range(args.epochs):
        model.train()
        epoch_loss=0.
        for imgs,masks in train_loader:
            imgs,masks = imgs.to(device),masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)["out"]
            loss = dice_loss(out,masks)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs,masks in val_loader:
                imgs,masks = imgs.to(device),masks.to(device)
                pred = (torch.sigmoid(model(imgs)["out"])>0.5).float()
                inter =(pred*masks).sum(dim=(2,3))
                union = pred.sum(dim=(2,3))+masks.sum(dim=(2,3))
                val_dice += ((2*inter+1)/(union + 1)).mean().item()
        val_dice /=len(val_loader)
        print(f"[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.6f} "
              f"epoch_loss: {epoch_loss:.4f}  val_dice: {val_dice:.4f}")
        scheduler.step()
        if val_dice > best_dice:
            best_dice =val_dice
            torch.save(model.state_dict(),"save_weights/best_cosine.pth")
            print("保存最优模型")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
