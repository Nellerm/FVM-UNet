import time
import sys

from torch.utils.data import DataLoader

from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.fvmunet.fvmunet import FVMUNet


from engine import *

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    if len(config.gpu_ids) > 1:
        # Convert each GPU ID to string
        gpu_ids_str = [str(id) for id in config.gpu_ids]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids_str)
        print(f"Using GPUs: {', '.join(gpu_ids_str)} (Parallel Mode)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
        print(f"Using GPU: {config.gpu_ids[0]}")

    # set seed
    set_seed(config.seed) # in utils.py
    # Clear cache
    torch.cuda.empty_cache()


    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    print("datasets:",config.datasets)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    #----------------fvmunet-----------------------#
    if config.network == 'fvmunet':
        model = FVMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else:
        raise Exception('network in not right!')
    model = model.cuda()


    # DataParallel
    if len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)

    cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    max_miou = 0
    start_epoch = 1
    min_epoch = 1
    min_epoch_miou = 1






    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss, max_miou, min_epoch_miou = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss'], checkpoint['max_miou'], checkpoint['min_epoch_miou']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    # Track the start time of the training process
    start_time = time.time()
    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()  # Start time for the current epoch

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss, val_miou = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        if val_miou > max_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_mIou_model.pth'))
            max_miou = val_miou
            min_epoch_miou = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'max_miou': max_miou,
                'min_epoch_miou': min_epoch_miou,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        # Calculate the time taken for the epoch
        epoch_duration = time.time() - epoch_start_time
        average_epoch_time = (time.time() - start_time) / epoch  # Average time per epoch
        remaining_epochs = config.epochs - epoch
        estimated_time_remaining = average_epoch_time * remaining_epochs

        # Convert estimated time remaining to hours, minutes, seconds
        hours, remainder = divmod(estimated_time_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Epoch {epoch}/{config.epochs} completed in {epoch_duration:.2f} seconds")
        print(f"Estimated time remaining: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        # best_weight_miou = torch.load(config.work_dir + 'checkpoints/best_mIou_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        print_model_parameters(model, checkpoint_dir)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best_mIou_model.pth'),
            os.path.join(checkpoint_dir, f'best_miou-epoch{min_epoch_miou}-miou{max_miou:.4f}.pth')
        )



if __name__ == '__main__':
    config = setting_config
    main(config)