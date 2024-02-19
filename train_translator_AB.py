import argparse
import torch
from dconv_model import DistillNet
from ImageLoaders import PairedImageSet
from loss import PerceptualLossModule
from torch.optim.lr_scheduler import MultiStepLR  
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import analyze_image_pair, analyze_image_pair_rgb, compute_shadow_mask_otsu
import os  
import gc
from PIL import Image
from torchvision import transforms
import numpy as np
import wandb

os.environ['TORCH_HOME'] = "./loaded_models/"

if __name__ == '__main__':
    # parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
    parser.add_argument("--resume_epoch", type=int, default=1, help="epoch to resume training")  # 重载训练，从之前中断处接着
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--gamma", type=float, default=0.2, help="adam: 学习率衰减的乘数")

    parser.add_argument("--decay_epoch", type=int, default=8, help="epoch from which to start lr decay")
    parser.add_argument("--decay_steps", type=int, default=2, help="number of step decays")

    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_height", type=int, default=2048, help="size of image height")
    # parser.add_argument("--img_width", type=int, default=2048, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--pixelwise_weight", type=float, default=1.0, help="Pixelwise loss weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Perceptual loss weight")
    parser.add_argument("--mask_weight", type=float, default=0.05, help="mask loss weight")

    parser.add_argument("--valid_checkpoint", type=int, default=1, help="checkpoint for validation")
    parser.add_argument("--save_checkpoint", type=int, default=2, help="checkpoint for visual inspection")
    opt = parser.parse_args()

    wandb.init(project="DNSR+MaterialData", config=vars(opt))
    wandb.config.update(opt)

    print('CUDA: ', torch.cuda.is_available(), torch.cuda.device_count())

    criterion_pixelwise = torch.nn.MSELoss() 
    pl = PerceptualLossModule()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    translator = DistillNet(num_iblocks=6, num_ops=4)
    translator.load_state_dict(torch.load("./loaded_models/gen_sh2f.pth"))
    translator = translator.to(device)
      
    print("USING CUDA FOR MODEL TRAINING")
    translator.cuda()
    criterion_pixelwise.cuda()

    # optimizer_G.load_state_dict(torch.load("./loaded_models/dnsr_wsrd_checkpoint/optimizer_ABG.pth"))
    optimizer_G = torch.optim.Adam(translator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    decay_step = (opt.n_epochs - opt.decay_epoch) // opt.decay_steps
    milestones = [me for me in range(opt.decay_epoch, opt.n_epochs, decay_step)] 
    scheduler = MultiStepLR(optimizer_G, milestones=milestones, gamma=opt.gamma)
   
    Tensor = torch.cuda.FloatTensor

    train_set = PairedImageSet('./dataset', 'train', use_mask=False, aug=False)
                               # size=(opt.img_height, opt.img_width)
    validation_set = PairedImageSet('./dataset', 'validation', use_mask=False, aug=False)

    dataloader = DataLoader(
        train_set,  
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,  
        drop_last=True  
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
    )

    num_samples = len(dataloader)
    val_samples = len(val_dataloader)

    best_rmse = 600
    # valid_table_data = []
    # train_table_data = []

    wandb.define_metric("Epoch")
    wandb.define_metric("train/*", step_metric="Epoch")
    wandb.define_metric("valid/*", step_metric="Epoch")
    wandb.define_metric("err/*", step_metric="Epoch")

    # wandb.define_metric("train/loss_epoch", summary="min")
    # wandb.define_metric("err/rmse_epoch", summary="min")
        
    for epoch in range(opt.resume_epoch, opt.n_epochs):
        train_epoch_loss = 0
        train_epoch_pix_loss = 0
        train_epoch_perc_loss = 0
        train_epoch_mask_loss = 0

        valid_epoch_loss = 0
        valid_mask_loss = 0
        valid_perc_loss = 0
        valid_pix_loss = 0

        err_epoch = 0
        err_rmse_epoch = 0
        err_psnr_epoch = 0

        translator = translator.cuda()
        translator = translator.train()

        for i, (B_img, AB_mask, A_img) in enumerate(dataloader):
            B_img = B_img.to(device)
            AB_mask = AB_mask.to(device)
            A_img = A_img.to(device)

            # 将图像分割为16个512x512块
            for m in range(4):
                for n in range(4):
                    left = n * 512
                    upper = m * 512
                    right = (n + 1) * 512
                    lower = (m + 1) * 512

                    gt = B_img[:, :, upper:lower, left:right]
                    mask = AB_mask[:, :, upper:lower, left:right]
                    inp = A_img[:, :, upper:lower, left:right]

                    # 将每个块送入网络模型进行训练,输出结果     
                    optimizer_G.zero_grad()
                    out = translator(inp, mask)
                    # mask计算中使用的otsu方法计算阴影遮罩mask不太靠谱吧。。。。
                        
                    # 模仿源文件，设计一系列loss计算
                    synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                    mask_loss = criterion_pixelwise(synthetic_mask, mask)
                    loss_pixel = criterion_pixelwise(out, gt)
                    perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                    loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss
                    loss_G.backward()  # 计算/累积梯度
                        
                    # 计算每一块的tile_loss之和，遍历所有pic的所有16 tiles
                    train_epoch_loss += loss_G.detach().item()
                    train_epoch_pix_loss += loss_pixel.detach().item()
                    train_epoch_perc_loss += perceptual_loss.detach().item()
                    train_epoch_mask_loss += mask_loss.detach().item()

            # 一个batch后更新模型参数
            optimizer_G.step()

        # train_table_data.append([epoch, train_epoch_loss, train_epoch_pix_loss, train_epoch_perc_loss, train_epoch_mask_loss]) # 一个epoch结束
        
        wandb.log({  # log是画出曲线图
             "train/loss_epoch": train_epoch_loss,
             "train/mask_loss_epoch": train_epoch_mask_loss,
             "train/pix_loss_epoch": train_epoch_pix_loss,
             "train/perc_loss_epoch": train_epoch_perc_loss,
             "Epoch": epoch
         })

        scheduler.step() # 训练结束，根据设定更新学习率

# 在评估阶段，模型的目标是衡量其在未见过的数据上的性能，而不再进行参数的更新。这通常发生在训练完成后，用于验证集或测试集上。
# 在评估阶段，模型应该是固定的，不再进行参数更新。为了获得一致的结果，一些层（如 Batch Normalization）可能会使用固定的统计数据而不是每个批次的统计数据。
        if epoch % opt.valid_checkpoint == 0:
            with torch.no_grad():
                translator = translator.eval()

                for idx, (B_img, AB_mask, A_img) in enumerate(val_dataloader):
                    B_img = B_img.to(device)
                    AB_mask = AB_mask.to(device)
                    A_img = A_img.to(device)

                    if epoch % opt.save_checkpoint == 0:
                        all_tiles = []

                    # 将图像分割为 16 个 512x512 的块
                    for m in range(4):
                        for n in range(4):
                            left = n * 512
                            upper = m * 512
                            right = (n + 1) * 512
                            lower = (m + 1) * 512

                            gt = B_img[:, :, upper:lower, left:right]
                            mask = AB_mask[:, :, upper:lower, left:right]
                            inp = A_img[:, :, upper:lower, left:right]

                            # 将每个块送入网络模型进行训练,输出结果
                            optimizer_G.zero_grad()
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = translator(inp, mask)

                            if epoch % opt.save_checkpoint == 0:
                                out_img = transforms.ToPILImage()(out[0])
                                all_tiles.append(out_img)
                                if m == 3 and n == 3:
                                    fullout = Image.new('RGB', (2048, 2048)) # 创建一个空白的完整图片
                                    for i, tile in enumerate(all_tiles): # 遍历all_tiles中的小块，将它们拼接到完整图片上
                                        row = i // 4
                                        col = i % 4
                                        left = col * 512
                                        upper = row * 512
                                        fullout.paste(tile, (left, upper)) # 此处更改为接缝处理算法！！
                                    wandb.log({"savepoint_fullout_epoch{}_{}".format(epoch, idx): [wandb.Image(fullout)]})

                            # 模仿源文件，设计一系列loss计算
                            synthetic_mask = compute_shadow_mask_otsu(inp, out.clone().detach())
                            mask_loss = criterion_pixelwise(synthetic_mask, mask)
                            loss_pixel = criterion_pixelwise(out, gt)
                            perceptual_loss = pl.compute_perceptual_loss_v(out.detach(), gt.detach())
                            loss_G = opt.pixelwise_weight * loss_pixel + opt.perceptual_weight * perceptual_loss + opt.mask_weight * mask_loss

                            rmse, psnr = analyze_image_pair_rgb(out.squeeze(0), gt.squeeze(0))
                            re, _ = analyze_image_pair(out.squeeze(0), gt.squeeze(0))

                            # 计算每一块的tile_loss之和，遍历所有val_pic的所有16 tiles
                            valid_epoch_loss += loss_G.detach().item()
                            valid_mask_loss += mask_loss.detach().item()
                            valid_pix_loss += loss_pixel.detach().item()
                            valid_perc_loss += perceptual_loss.detach().item()

                            err_epoch += re
                            err_rmse_epoch += rmse
                            err_psnr_epoch += psnr

                # err_epoch /= val_samples
                # err_rmse_epoch /= val_samples
                # err_psnr_epoch /= val_samples

                # valid_table_data.append(
                #     [epoch, valid_epoch_loss, valid_pix_loss, valid_perc_loss, valid_mask_loss, err_epoch, err_rmse_epoch, err_psnr_epoch])

                wandb.log({
                     "valid/loss_epoch": valid_epoch_loss,
                     "valid/mask_loss_epoch": valid_mask_loss,
                     "valid/pix_loss_epoch": valid_pix_loss,
                     "valid/perc_loss_epoch": valid_perc_loss,
                     "err/epoch":  err_epoch,
                     "err/rmse_epoch":  err_rmse_epoch,
                     "err/psnr_epoch":  err_psnr_epoch,
                     "Epoch": epoch
                })

        print("EPOCH: {} - LOSS: {:.3f} | {:.3f} - MskLoss: {:.3f} | {:.3f} - RMSE {:.3f} - PSNR - {:.3f}".format(
                                                                                    epoch, train_epoch_loss,
                                                                                    valid_epoch_loss, train_epoch_mask_loss,
                                                                                    valid_mask_loss,
                                                                                    err_rmse_epoch,  err_psnr_epoch))
        
        if err_rmse_epoch < best_rmse and epoch > 1:
            best_rmse = err_rmse_epoch
            print("Saving checkpoint for epoch {} and RMSE {}".format(epoch, best_rmse))
            torch.save(translator.cpu().state_dict(), "./best_rmse_model/distillnet_epoch{}.pth".format(epoch))
            torch.save(optimizer_G.state_dict(), "./best_rmse_model/optimizer_epoch{}.pth".format(epoch))
            wandb.config.update({"best_rmse": best_rmse}, allow_val_change=True)


    # train_table = wandb.Table(data=train_table_data, columns=["Epoch", "Train_Epoch_Loss", "Train_Epoch_Pix_Loss", "Train_Epoch_Perc_Loss", "Train_Epoch_Mask_Loss"])
    # wandb.log({"Train Epoch Loss Table": train_table})
    # valid_table = wandb.Table(data=valid_table_data, columns=["Epoch", "Valid_Epoch_Loss", "Valid_Epoch_Pix_Loss",
    #                                                     "Valid_Epoch_Perc_Loss", "Valid_Epoch_Mask_Loss",
    #                                                     "Err_Epoch", "Err_RMSE_Epoch", "Err_PSNR_Epoch"])
    # wandb.log({"Valid Epoch Loss&Error Table": valid_table})
