import math
from math import sqrt
import argparse
import os
import cv2
import time
import logging
import sys
import random
import numpy as np
import warnings
from typing import Callable, cast
import util
# torch

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast

from torchnet.dataset import SplitDataset

from einops import rearrange
from torchvision import transforms as T
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from albumentations import (
    RandomResizedCrop, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE,
    RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise,
    MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Resize, VerticalFlip,
    HorizontalFlip, CenterCrop, Normalize)

# dalle classes and utils

from dalle_pytorch import deepspeed_utils
from dalle_pytorch import DiscreteVAE, ResBlock

try:
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch.state_mod import state as smp_state
except ImportError:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class AlbumentationImageDataset(Dataset):
    def __init__(self, image_path, transform, args):
        self.image_path = image_path
        self.transform = transform
        self.args = args
        self.image_list = self._loader_file(self.image_path)

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        
        image = cv2.imread(self.image_list[i][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augment an image
        transformed = self.transform(image=image)["image"]
        transformed_image = np.transpose(transformed,
                                         (2, 0, 1)).astype(np.float32)
        return torch.tensor(transformed_image,
                            dtype=torch.float), self.image_list[i][1]

    def _loader_file(self, image_path):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                      '.tiff', '.webp')
        
        def is_valid_file(x: str) -> bool:
            return x.lower().endswith(extensions)

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        self.classes = [d.name for d in os.scandir(image_path) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(self.classes)
        }

        instances = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(image_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir,
                                                  followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        return instances

    
# argument parsing
def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--image_folder', type = str, required = True,
    #                     help='path to your folder of images for learning the discrete VAE and its codebook')

    
    ## Data/Model/Output
    parser.add_argument('--image_folder', type = str, default = '../../dataset/val2017')
    parser.add_argument('--model_dir', type=str, default='../model/vae') 
    parser.add_argument('--output_dir', type=str, default='../output/vae') 
    parser.add_argument('--image_size', type = int, required = False, default = 256,
                        help='image size')
    ## Hyperparameter
    parser.add_argument('--EPOCHS', type=int, default=20)
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-3)
    parser.add_argument('--LR_DECAY_RATE', type=float, default=0.98)

    parser.add_argument('--NUM_TOKENS', type=int, default=8192)
    parser.add_argument('--NUM_LAYERS', type=int, default=2)
    parser.add_argument('--NUM_RESNET_BLOCKS', type=int, default=2)
    parser.add_argument('--SMOOTH_L1_LOSS', type=bool, default=False)
    parser.add_argument('--EMB_DIM', type=int, default=512)
    parser.add_argument('--HID_DIM', type=int, default=256)
    parser.add_argument('--KL_LOSS_WEIGHT', type=int, default=0)

    parser.add_argument('--STARTING_TEMP', type=float, default=1.)
    parser.add_argument('--TEMP_MIN', type=float, default=0.5)
    parser.add_argument('--ANNEAL_RATE', type=float, default=1e-6)

    parser.add_argument('--NUM_IMAGES_SAVE', type=int, default=4)
    parser.add_argument('--model_parallel', type=bool, default=False)    
    parser.add_argument('--num_worker', type=int, default=4)  

    # Setting for Model Parallel
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--num_partitions", type=int, default=2)
    parser.add_argument("--horovod", type=bool, default=False)
    parser.add_argument("--ddp", type=bool, default=True)
    parser.add_argument("--amp", type=int, default=0)  ## if amp is 1, true 
    parser.add_argument("--pipeline", type=str, default="interleaved")
    parser.add_argument("--optimize", type=str, default="speed")
    parser.add_argument("--placement_strategy", type=str, default="spread")
    parser.add_argument("--assert-losses", type=bool, default=False)
    
    parser.add_argument('--mp_parameters', type=str, default='')
    parser.add_argument("--partial-checkpoint",
                        type=str,
                        default="",
                        help="The checkpoint path to load")
    parser.add_argument("--full-checkpoint",
                        type=str,
                        default="",
                        help="The checkpoint path to load")
    parser.add_argument("--save-full-model",
                        action="store_true",
                        default=False,
                        help="For Saving the current Model")
    parser.add_argument(
        "--save-partial-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument('--hosts',
                        type=list,
                        default=['algo-1'])
    parser.add_argument('--num-gpus',
                        type=int,
                        default=4)
    parser.add_argument('--channels-last', type=bool, default=True)
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=5,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser = deepspeed_utils.wrap_arg_parser(parser)
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise ValueError("The script requires CUDA support, but CUDA not available")
        
    args.rank = -1
    args.world_size = 1
    
    if args.model_parallel:
        args.deepspeed = False
        cfg = {
            "microbatches": args.num_microbatches,
            "placement_strategy": args.placement_strategy,
            "pipeline": args.pipeline,
            "optimize": args.optimize,
            "partitions": args.num_partitions,
            "horovod": args.horovod,
            "ddp": args.ddp,
        }

        smp.init(cfg)    
        torch.cuda.set_device(smp.local_rank())
        args.rank = smp.dp_rank()
        args.world_size = smp.size()
    else:
        # initialize deepspeed
        print(f"args.deepspeed : {args.deepspeed}")
        deepspeed_utils.init_deepspeed(args.deepspeed)
        if deepspeed_utils.is_root_worker():
            args.rank = 0
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed+args.rank)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        
    # args.LEARNING_RATE = args.LEARNING_RATE * float(args.world_size)
    
    cudnn.deterministic = True

    if cudnn.deterministic:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    args.kwargs = {
        'num_workers': args.num_worker,
        'pin_memory': True
    }
    
    device = torch.device("cuda")
    
    logger.debug(f"args.image_folder : {args.image_folder}")
    logger.debug(f"args.rank : {args.rank}")
    

    ## SageMaker
    try:
        if os.environ.get('SM_MODEL_DIR') is not None:
            args.model_dir = os.environ.get('SM_MODEL_DIR')
            args.output_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
            args.image_folder = os.environ.get('SM_CHANNEL_TRAINING')
    except:
        logger.debug("not SageMaker")
        pass
    
    IMAGE_SIZE = args.image_size
    IMAGE_PATH = args.image_folder

    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    LEARNING_RATE = args.LEARNING_RATE
    LR_DECAY_RATE = args.LR_DECAY_RATE

    NUM_TOKENS = args.NUM_TOKENS
    NUM_LAYERS = args.NUM_LAYERS
    NUM_RESNET_BLOCKS = args.NUM_RESNET_BLOCKS
    SMOOTH_L1_LOSS = args.SMOOTH_L1_LOSS
    EMB_DIM = args.EMB_DIM
    HID_DIM = args.HID_DIM
    KL_LOSS_WEIGHT = args.KL_LOSS_WEIGHT

    STARTING_TEMP = args.STARTING_TEMP
    TEMP_MIN = args.TEMP_MIN
    ANNEAL_RATE = args.ANNEAL_RATE

    NUM_IMAGES_SAVE = args.NUM_IMAGES_SAVE


    
#     transform = Compose(
#         [
#             RandomResizedCrop(args.image_size, args.image_size),
#             OneOf(
#                 [
#                     IAAAdditiveGaussianNoise(),
#                     GaussNoise(),
#                 ], 
#                 p=0.2
#             ),
#             VerticalFlip(p=0.5),
#             OneOf(
#                 [
#                     MotionBlur(p=.2),
#                     MedianBlur(blur_limit=3, p=0.1),
#                     Blur(blur_limit=3, p=0.1),
#                 ],
#                 p=0.2
#             ),
#             OneOf(
#                 [
#                     CLAHE(clip_limit=2),
#                     IAASharpen(),
#                     IAAEmboss(),
#                     RandomBrightnessContrast(),
#                 ],
#                 p=0.3
#             ),
#             HueSaturationValue(p=0.3),
# #             Normalize(
# #                 mean=[0.485, 0.456, 0.406],
# #                 std=[0.229, 0.224, 0.225],
# #             )
#         ],
#         p=1.0
#     )
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor()
    ])
    
    sampler = None
    dl = None

    # data
    logger.debug(f"IMAGE_PATH : {IMAGE_PATH}")
#     ds = AlbumentationImageDataset(
#         IMAGE_PATH,
#         transform=transform,
#         args=args
#     )
    ds = ImageFolder(
        IMAGE_PATH,
        transform=transform,
    )
    
    if args.model_parallel and (args.ddp or args.horovod) and smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        ds = SplitDataset(ds, partitions=partitions_dict)
        ds.select(f"{smp.dp_rank()}")

    dl = DataLoader(ds, BATCH_SIZE, 
                    shuffle=True,
                    drop_last=args.model_parallel,
                    **args.kwargs)


    vae_params = dict(
        image_size = IMAGE_SIZE,
        num_layers = NUM_LAYERS,
        num_tokens = NUM_TOKENS,
        codebook_dim = EMB_DIM,
        hidden_dim   = HID_DIM,
        num_resnet_blocks = NUM_RESNET_BLOCKS
    )

    vae = DiscreteVAE(
        **vae_params,
        smooth_l1_loss = SMOOTH_L1_LOSS,
        kl_div_loss_weight = KL_LOSS_WEIGHT
    ).to(device)
    # optimizer 
    
    opt = Adam(vae.parameters(), lr = LEARNING_RATE)
    sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)
    
    if args.model_parallel:
        import copy
        dummy_codebook = copy.deepcopy(vae.codebook)
        dummy_decoder = copy.deepcopy(vae.decoder)        
        
        vae = smp.DistributedModel(vae)
        scaler = smp.amp.GradScaler()
        opt = smp.DistributedOptimizer(opt)
        
        if args.partial_checkpoint:
            args.checkpoint = smp.load(args.partial_checkpoint, partial=True)
            vae.load_state_dict(args.checkpoint["model_state_dict"])
            opt.load_state_dict(args.checkpoint["optimizer_state_dict"])
        elif args.full_checkpoint:
            args.checkpoint = smp.load(args.full_checkpoint, partial=False)
            vae.load_state_dict(args.checkpoint["model_state_dict"])
            opt.load_state_dict(args.checkpoint["optimizer_state_dict"])

 
    assert len(ds) > 0, 'folder does not contain any images'
    
    if (not args.model_parallel) and args.rank == 0:
        print(f'{len(ds)} images found for training')
        
        # weights & biases experiment tracking

#         import wandb

        model_config = dict(
            num_tokens = NUM_TOKENS,
            smooth_l1_loss = SMOOTH_L1_LOSS,
            num_resnet_blocks = NUM_RESNET_BLOCKS,
            kl_loss_weight = KL_LOSS_WEIGHT
        )

#         run = wandb.init(
#             project = 'dalle_train_vae',
#             job_type = 'train_model',
#             config = model_config
#         )

    def save_model(path):
        if not args.rank == 0:
            return

        save_obj = {
            'hparams': vae_params,
            'weights': vae.state_dict()
        }

        torch.save(save_obj, path)


    # distribute with deepspeed
    if not args.model_parallel:
        deepspeed_utils.check_batch_size(BATCH_SIZE)
        deepspeed_config = {'train_batch_size': BATCH_SIZE}

        (distr_vae, opt, dl, sched) = deepspeed_utils.maybe_distribute(
            args=args,
            model=vae,
            optimizer=opt,
            model_parameters=vae.parameters(),
            training_data=ds if args.deepspeed else dl,
            lr_scheduler=sched,
            config_params=deepspeed_config,
        )
        
        
    try:
        # Rubik: Define smp.step. Return any tensors needed outside.
        @smp.step
        def train_step(vae, images, temp):
#             logger.debug(f"args.amp : {args.amp}")
            with autocast(enabled = (args.amp > 0)):
                loss, recons = vae(
                    images,
                    return_loss = True,
                    return_recons = True,
                    temp = temp)
            
            scaled_loss = scaler.scale(loss) if args.amp else loss
            vae.backward(scaled_loss)
#             torch.nn.utils.clip_grad_norm_(vae.parameters(), 5)
            return loss, recons
        
        @smp.step
        def get_codes_step(vae, images, k):
            images = images[:k]
            logits = vae.forward(images, return_logits = True)
            codebook_indices = logits.argmax(dim = 1).flatten(1)
            return codebook_indices

        
        def hard_recons_step(dummy_decoder, dummy_codebook, codebook_indices):
            from functools import partial
            for module in dummy_codebook.modules():
                method = smp_state.patch_manager.get_original_method("forward", type(module))
                module.forward = partial(method, module)
            image_embeds = dummy_codebook.forward(codebook_indices)
            b, n, d = image_embeds.shape
            h = w = int(sqrt(n))
            
            image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
            for module in dummy_decoder.modules():
                method = smp_state.patch_manager.get_original_method("forward", type(module))
                module.forward = partial(method, module)
            hard_recons = dummy_decoder.forward(image_embeds)
            return hard_recons

    except:
        pass        
    
    # starting temperature

    global_step = 0
    temp = STARTING_TEMP

    
    for epoch in range(EPOCHS):
        ##
        batch_time = util.AverageMeter('Time', ':6.3f')
        data_time = util.AverageMeter('Data', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        top1 = util.AverageMeter('Acc@1', ':6.2f')
        top5 = util.AverageMeter('Acc@5', ':6.2f')
        progress = util.ProgressMeter(
            len(dl), [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        vae.train()
        start = time.time()
        
        for i, (images, _) in enumerate(dl):
            images = images.to(device, non_blocking=True)
            opt.zero_grad()
            
            if args.model_parallel:
                loss, recons = train_step(vae, images, temp)
                # Rubik: Average the loss across microbatches.
                loss = loss.reduce_mean()
                recons = recons.reduce_mean()
            else:
                loss, recons = distr_vae(
                    images,
                    return_loss = True,
                    return_recons = True,
                    temp = temp
                )
            
            if (not args.model_parallel) and args.deepspeed:
                # Gradients are automatically zeroed after the step
                distr_vae.backward(loss)
                distr_vae.step()
            elif args.model_parallel:
                if args.amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    # some optimizers like adadelta from PT 1.8 dont like it when optimizer.step is called with no param
                    if len(list(vae.local_parameters())) > 0:
                        opt.step()
            else:
                loss.backward()
                opt.step()

            logs = {}

            if i % 10 == 0:
                if args.rank == 0:
#                 if deepspeed_utils.is_root_worker():
                    k = NUM_IMAGES_SAVE
                    
                    with torch.no_grad():
                        if args.model_parallel:
                            model_dict = vae.state_dict()
                            model_dict_updated = {}
                            for key, val in model_dict.items():
                                if "decoder" in key:
                                    key = key.replace("decoder.", "")
                                elif "codebook" in key:
                                    key = key.replace("codebook.", "")
                                model_dict_updated[key] = val
                                    
                            dummy_decoder.load_state_dict(model_dict_updated, strict=False)
                            dummy_codebook.load_state_dict(model_dict_updated, strict=False)
                            codes = get_codes_step(vae, images, k)
                            codes = codes.reduce_mean().to(torch.long)
                            hard_recons = hard_recons_step(dummy_decoder, dummy_codebook, codes)
                        else:
                            codes = vae.get_codebook_indices(images[:k])
                            hard_recons = vae.decode(codes)

                    images, recons = map(lambda t: t[:k], (images, recons))
                    images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
                    images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))

#                     logs = {
#                         **logs,
#                         'sample images':        wandb.Image(images, caption = 'original images'),
#                         'reconstructions':      wandb.Image(recons, caption = 'reconstructions'),
#                         'hard reconstructions': wandb.Image(hard_recons, caption = 'hard reconstructions'),
#                         'codebook_indices':     wandb.Histogram(codes),
#                         'temperature':          temp
#                     }
  
                if args.model_parallel:
                    filename = f'{args.output_dir}/vae.pt'
                    if smp.dp_rank == 0:
                        if args.save_full_model:
                            model_dict = vae.state_dict()
                            opt_dict = opt.state_dict()
                            smp.save(
                                {
                                    "model_state_dict": model_dict,
                                    "optimizer_state_dict": opt_dict
                                },
                                filename,
                                partial=False,
                            )
                        else:
                            model_dict = vae.local_state_dict()
                            opt_dict = opt.local_state_dict()
                            smp.save(
                                {
                                    "model_state_dict": model_dict,
                                    "optimizer_state_dict": opt_dict
                                },
                                filename,
                                partial=True,
                            )
                    smp.barrier()
                    
                   
                else:
                    save_model(f'{args.output_dir}/vae.pt')
    #                     wandb.save(f'{args.output_dir}/vae.pt')

                # temperature anneal

                temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

                # lr decay

                sched.step()

            # Collective loss, averaged
            if args.model_parallel:
                avg_loss = loss.detach().clone()
#                 print("args.world_size : {}".format(args.world_size))
                avg_loss /= args.world_size
        
            
            else:
                avg_loss = deepspeed_utils.average_all(loss)
            

            if args.rank == 0:
                if i % 100 == 0:
                    lr = sched.get_last_lr()[0]
                    print(epoch, i, f'lr - {lr:6f}, loss - {avg_loss.item()},')

                    logs = {
                        **logs,
                        'epoch': epoch,
                        'iter': i,
                        'loss': avg_loss.item(),
                        'lr': lr
                    }

#                 wandb.log(logs)
            global_step += 1
    
            if args.rank == 0:
                # Every print_freq iterations, check the loss, accuracy, and speed.
                # For best performance, it doesn't make sense to print these metrics every
                # iteration, since they incur an allreduce and some host<->device syncs.


                # Measure accuracy
#                 prec1, prec5 = util.accuracy(output, target, topk=(1, 5))

                # to_python_float incurs a host<->device sync
                losses.update(util.to_python_float(loss), images.size(0))
#                 top1.update(util.to_python_float(prec1), images.size(0))
#                 top5.update(util.to_python_float(prec5), images.size(0))

                # Waiting until finishing operations on GPU (Pytorch default: async)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / args.log_interval)
                end = time.time()

                print(
                    'Epoch: [{0}][{1}/{2}] '
                    'Train_Time={batch_time.val:.3f}: avg-{batch_time.avg:.3f}, '
                    'Train_Speed={3:.3f} ({4:.3f}), '
                    'Train_Loss={loss.val:.10f}:({loss.avg:.4f}),'.format(
                        epoch,
                        i,
                        len(dl),
                        args.world_size * BATCH_SIZE / batch_time.val,
                        args.world_size * BATCH_SIZE / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses))

#         if deepspeed_utils.is_root_worker():
            # save trained model to wandb as an artifact every epoch's end

#             model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
#             model_artifact.add_file(f'{args.model_dir}/vae.pt')
#             run.log_artifact(model_artifact)

    if args.rank == 0:
#     if deepspeed_utils.is_root_worker():
        # save final vae and cleanup
        if args.model_parallel:
            logger.debug('save model_parallel')
        else:
            save_model(os.path.join(args.model_dir, 'vae-final.pt'))
#         wandb.save(f'{args.model_dir}/vae-final.pt')

#         model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
#         model_artifact.add_file(f'{args.model_dir}/vae-final.pt')
#         run.log_artifact(model_artifact)

#         wandb.finish()



    if args.model_parallel:
        if args.assert_losses:
            if args.horovod or args.ddp:
                # SM Distributed: If using data parallelism, gather all losses across different model
                # replicas and check if losses match.

                losses = smp.allgather(loss, smp.DP_GROUP)
                for l in losses:
                    print(l)
                    assert math.isclose(l, losses[0])

                assert loss < 0.18
            else:
                assert loss < 0.08

        smp.barrier()
        print("SMP training finished successfully")

if __name__ == "__main__":
    main()