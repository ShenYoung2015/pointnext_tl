from Parameters import *
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.dirname(os.path.abspath(__name__)))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from Model import PointNeXt
from dataset.FFDshape import FFDshape_ptp_tl
from Loss import LabelSmoothingCE, reg_loss
from Transforms import PCDPretreatment, get_data_augment
from Trainer import Trainer
from utils import IdentityScheduler
import numpy as np

def main():
    args.dataset = 'waverider'
    if args.dataset == 'waverider':
        default_dataset_path_list = [r'I:/PCdeep_TL/shapesffd4/waverider/shapes_N50_D60_322']
    checkpoint_name = 'PointNeXt_shapesffd3_epoch1000.pth'
    checkpoint_dir = 'result_train//PointNeXt_model=basic_c_ds=shapesffd3_aug=basic_lr=0.001_wd=0.0001_bs=16_AdamW_cosine//'
    checkpoint_file = checkpoint_dir + checkpoint_name
    
    need_frozen_list_path = 'need_frozen_list.csv'
    need_frozen_np = np.loadtxt(need_frozen_list_path, dtype=str, delimiter=',')
    need_frozen_list = {}
    for i, a in enumerate(need_frozen_np[:,0]):
        need_frozen_list[a] = need_frozen_np[i,1]

    # 解析参数
    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')
    
    model_cfg = MODEL_CONFIG[args.model_cfg]
    max_input = model_cfg['max_input']
    normal = model_cfg['normal']
    
    if args.optimizer.lower() == 'adamw':
        Optimizer = torch.optim.AdamW
    else:
        args.optimizer = 'Adam'
        Optimizer = torch.optim.Adam

    if args.scheduler.lower() == 'identity':
        Scheduler = IdentityScheduler
    else:
        args.scheduler = 'cosine'
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    # 数据变换、加载数据集
    logger.info('Prepare Data')
    '''数据变换、加载数据集'''
    data_augment, random_sample, random_drop = get_data_augment(DATA_AUG_CONFIG[args.data_aug])
    transforms = PCDPretreatment(num=max_input, down_sample='random', normal=normal,
                                 data_augmentation=data_augment, random_drop=random_drop, resampling=random_sample)


        
    for path in default_dataset_path_list:
        if os.path.exists(path):
            args.dataset_path = path
            break
    else:  # this is for-else block, indent is not missing
        raise FileNotFoundError(f'Dataset path not found.')
    logger.info(f'Load default dataset from {args.dataset_path}')

    dataset = FFDshape_ptp_tl(root=args.dataset_path, transforms=transforms)

    # 模型与损失函数
    logger.info('Prepare Models...')
    model = PointNeXt(model_cfg).to(device=args.device)
    checkpoint = torch.load(checkpoint_file, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.train()
    # frozen
    for param in model.named_parameters():
        if need_frozen_list[param[0]] == '1':
            # frozen
            param[1].requires_grad = False
        else:
            param[1].requires_grad = True
    # print(param[0],param[1].requires_grad)
    # args.lr = 1e-4
    # args.eval_cycle = 10
    # args.batch_size = 4

    # optimizer = Optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = Scheduler(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.001)
    criterion = reg_loss().to(args.device)


    # 训练器
    logger.info('Trainer launching...')
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataset=dataset,
        mode=args.mode
    )
    trainer.run()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
    print('Done.')
